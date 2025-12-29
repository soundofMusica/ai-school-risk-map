
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Optional dependency for drawing
try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# PDF export
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.units import mm


st.set_page_config(page_title="AI 학교 위험지도", layout="wide")

APP_DIR = Path(__file__).resolve().parent

DEFAULT_EXCEL = APP_DIR / "AI_학교위험지도_구역_체크리스트.xlsx"
DEFAULT_MAP = APP_DIR / "학교_지도.png"
DEFAULT_POLYGONS = APP_DIR / "polygons.json"


# -------------------------
# Helpers
# -------------------------
def safe_read_excel(xlsx_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(xlsx_path)
    # Try common sheet names
    zone_sheet = None
    checklist_sheet = None
    for name in xls.sheet_names:
        low = name.lower()
        if "zone" in low:
            zone_sheet = name
        if "checklist" in low:
            checklist_sheet = name
    if zone_sheet is None:
        zone_sheet = xls.sheet_names[0]
    if checklist_sheet is None:
        checklist_sheet = xls.sheet_names[-1] if len(xls.sheet_names) > 1 else xls.sheet_names[0]

    zones = pd.read_excel(xlsx_path, sheet_name=zone_sheet)
    checklist = pd.read_excel(xlsx_path, sheet_name=checklist_sheet)

    # Normalize expected columns for zones
    zones = zones.copy()
    # expected columns: zone_id, floor, zone_type, display_name, map_hint, notes, polygon_points
    col_map = {}
    for c in zones.columns:
        lc = str(c).strip().lower()
        if lc in ["zone_id", "id"]:
            col_map[c] = "zone_id"
        elif "floor" in lc or lc in ["층", "층수"]:
            col_map[c] = "floor"
        elif "type" in lc or "구역타입" in lc or lc in ["zone_type"]:
            col_map[c] = "zone_type"
        elif "display" in lc or "표시" in lc or "name" == lc or "이름" in lc:
            col_map[c] = "display_name"
        elif "hint" in lc or "지도" in lc:
            col_map[c] = "map_hint"
        elif "note" in lc or "메모" in lc:
            col_map[c] = "notes"
        elif "polygon" in lc or "points" in lc:
            col_map[c] = "polygon_points"
    zones.rename(columns=col_map, inplace=True)

    if "zone_id" not in zones.columns:
        # Create a fallback
        zones["zone_id"] = [f"ZONE_{i:03d}" for i in range(len(zones))]
    if "floor" not in zones.columns:
        zones["floor"] = zones["zone_id"].astype(str).str.extract(r"F(\d+)").fillna(0).astype(int)
    if "zone_type" not in zones.columns:
        zones["zone_type"] = "UNKNOWN"
    if "display_name" not in zones.columns:
        zones["display_name"] = zones["zone_id"]
    for col in ["map_hint", "notes", "polygon_points"]:
        if col not in zones.columns:
            zones[col] = ""

    # Checklist normalize
    checklist = checklist.copy()
    c_map = {}
    for c in checklist.columns:
        lc = str(c).strip().lower()
        if "id" == lc or "체크" in lc:
            c_map[c] = "check_id"
        elif "카테고리" in lc or "category" in lc:
            c_map[c] = "category"
        elif "항목" in lc or "item" in lc:
            c_map[c] = "item"
        elif "이유" in lc or "why" in lc:
            c_map[c] = "why"
        elif "방법" in lc or "how" in lc:
            c_map[c] = "how"
        elif "증거" in lc or "evidence" in lc:
            c_map[c] = "evidence"
        elif "관련" in lc or "zone" in lc:
            c_map[c] = "related_zone_type"
    checklist.rename(columns=c_map, inplace=True)
    for col in ["check_id", "category", "item", "why", "how", "evidence", "related_zone_type"]:
        if col not in checklist.columns:
            checklist[col] = ""

    # Clean
    zones["floor"] = pd.to_numeric(zones["floor"], errors="coerce").fillna(0).astype(int)
    zones["zone_id"] = zones["zone_id"].astype(str)
    zones["zone_type"] = zones["zone_type"].astype(str)
    zones["display_name"] = zones["display_name"].astype(str)

    return zones, checklist


def load_polygons(json_path: Path) -> Dict[str, Any]:
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_polygons(json_path: Path, polygons: Dict[str, Any]) -> None:
    json_path.write_text(json.dumps(polygons, ensure_ascii=False, indent=2), encoding="utf-8")


def rect_to_poly(rect: Dict[str, float]) -> List[List[float]]:
    x = float(rect["x"])
    y = float(rect["y"])
    w = float(rect["width"])
    h = float(rect["height"])
    return [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]


def poly_stats(poly: List[List[float]]) -> Dict[str, float]:
    pts = np.array(poly, dtype=float)
    xs, ys = pts[:, 0], pts[:, 1]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    w = max(1e-6, xmax - xmin)
    h = max(1e-6, ymax - ymin)
    area = float(w * h)  # rectangle-like area proxy
    cx, cy = float(xs.mean()), float(ys.mean())
    aspect = float(w / h) if h > 1e-6 else 1.0
    return {"cx": cx, "cy": cy, "w": float(w), "h": float(h), "area": area, "aspect": aspect}


def value_to_color(v: float, vmin: float, vmax: float) -> Tuple[int, int, int, int]:
    # Simple red colormap with alpha
    if vmax <= vmin:
        t = 0.0
    else:
        t = (v - vmin) / (vmax - vmin)
        t = float(np.clip(t, 0.0, 1.0))
    r = int(255 * t + 80 * (1 - t))
    g = int(80 * (1 - t))
    b = int(80 * (1 - t))
    a = int(140)  # alpha
    return (r, g, b, a)


def draw_overlay(img: Image.Image, zones: pd.DataFrame, polygons: Dict[str, Any], score_col: str) -> Image.Image:
    base = img.convert("RGBA").copy()
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # Gather valid scores
    valid = zones[score_col].dropna()
    if len(valid) == 0:
        vmin, vmax = 0.0, 100.0
    else:
        vmin, vmax = float(valid.min()), float(valid.max())

    for _, row in zones.iterrows():
        zid = row["zone_id"]
        if zid not in polygons:
            continue
        poly = polygons[zid].get("poly")
        if not poly:
            continue
        try:
            pts = [(float(x), float(y)) for x, y in poly]
        except Exception:
            continue
        v = row.get(score_col, np.nan)
        if pd.isna(v):
            continue
        color = value_to_color(float(v), vmin, vmax)
        d.polygon(pts, fill=color, outline=(255, 255, 255, 200))

    combined = Image.alpha_composite(base, overlay)
    return combined


def ensure_label_columns(zones: pd.DataFrame) -> pd.DataFrame:
    zones = zones.copy()
    # Label columns the user can fill from survey/simulation
    if "risk_label" not in zones.columns:
        zones["risk_label"] = np.nan  # 0~100
    if "scenario" not in zones.columns:
        zones["scenario"] = "기본(정전)"
    # Optional feature inputs
    for col in ["survey_darkness", "survey_confusion", "survey_confidence", "sim_congestion"]:
        if col not in zones.columns:
            zones[col] = np.nan
    return zones


def build_feature_table(zones: pd.DataFrame, polygons: Dict[str, Any]) -> pd.DataFrame:
    zones = zones.copy()
    # Geometry-derived features
    geom = []
    for zid in zones["zone_id"].tolist():
        if zid in polygons and polygons[zid].get("poly"):
            stats = poly_stats(polygons[zid]["poly"])
        else:
            stats = {"cx": np.nan, "cy": np.nan, "w": np.nan, "h": np.nan, "area": np.nan, "aspect": np.nan}
        stats["zone_id"] = zid
        geom.append(stats)
    geom_df = pd.DataFrame(geom)
    zones = zones.merge(geom_df, on="zone_id", how="left")

    # Basic engineered features
    zones["floor"] = pd.to_numeric(zones["floor"], errors="coerce").fillna(0).astype(int)
    zones["has_polygon"] = zones["area"].notna().astype(int)

    # Feature set (keep small + explainable)
    keep = [
        "zone_id", "floor", "zone_type",
        "cx", "cy", "area", "aspect",
        "survey_darkness", "survey_confusion", "survey_confidence", "sim_congestion",
        "risk_label", "scenario"
    ]
    for c in keep:
        if c not in zones.columns:
            zones[c] = np.nan
    return zones[keep]


@dataclass
class TrainedModel:
    rf: Any
    ridge: Any
    preprocessor: Any
    feature_names: List[str]
    mae_cv: Optional[float] = None


def train_models(df: pd.DataFrame) -> Optional[TrainedModel]:
    # Train on rows with risk_label
    train_df = df[df["risk_label"].notna()].copy()
    if len(train_df) < 10:
        return None

    X = train_df.drop(columns=["risk_label"])
    y = train_df["risk_label"].astype(float)

    num_cols = ["floor", "cx", "cy", "area", "aspect", "survey_darkness", "survey_confusion", "survey_confidence", "sim_congestion"]
    cat_cols = ["zone_type", "scenario"]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop"
    )

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    ridge = Ridge(alpha=1.0, random_state=42)

    rf_pipe = Pipeline([("pre", pre), ("model", rf)])
    ridge_pipe = Pipeline([("pre", pre), ("model", ridge)])

    # CV (MAE)
    kf = KFold(n_splits=min(5, len(train_df)), shuffle=True, random_state=42)
    maes = []
    for tr, te in kf.split(train_df):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        rf_pipe.fit(Xtr, ytr)
        pred = rf_pipe.predict(Xte)
        maes.append(mean_absolute_error(yte, pred))
    mae_cv = float(np.mean(maes))

    # Fit final
    rf_pipe.fit(X, y)
    ridge_pipe.fit(X, y)

    # Get feature names for ridge explanation
    oh = ridge_pipe.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
    cat_feature_names = list(oh.get_feature_names_out(["zone_type", "scenario"]))
    feature_names = [
        "floor", "cx", "cy", "area", "aspect",
        "survey_darkness", "survey_confusion", "survey_confidence", "sim_congestion",
        *cat_feature_names
    ]
    return TrainedModel(rf=rf_pipe, ridge=ridge_pipe, preprocessor=pre, feature_names=feature_names, mae_cv=mae_cv)


def explain_zone(model: TrainedModel, row: pd.Series) -> List[Tuple[str, float]]:
    # Use ridge (linear) to compute contributions: coef * x
    # Build a single-row DF aligned with training columns
    single = row.to_frame().T.copy()
    X = single.drop(columns=["risk_label"], errors="ignore")
    # Transform using ridge pipeline
    pre = model.ridge.named_steps["pre"]
    Xt = pre.transform(X)
    # Ridge coefficients
    coefs = model.ridge.named_steps["model"].coef_
    # Contributions
    contrib = (Xt.toarray() if hasattr(Xt, "toarray") else Xt) * coefs
    contrib = contrib.flatten()
    pairs = list(zip(model.feature_names, contrib))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:6]


# -------------------------
# UI
# -------------------------
st.title("🏫 AI 활용 학교 내부 위험지도 (정전/재난 대응)")

with st.sidebar:
    st.header("데이터 불러오기")
    excel_up = st.file_uploader("구역/체크리스트 엑셀 업로드(.xlsx)", type=["xlsx"])
    map_up = st.file_uploader("지도 이미지 업로드(.png/.jpg)", type=["png", "jpg", "jpeg"])
    poly_up = st.file_uploader("구역 폴리곤 JSON 업로드(선택)", type=["json"])
    st.divider()
    st.caption("※ 2~3주 MVP용: 구역은 직접 사각형으로 찍어도 충분합니다.")

# Load data
if excel_up is not None:
    excel_path = Path("uploaded.xlsx")
    excel_path.write_bytes(excel_up.getvalue())
else:
    excel_path = DEFAULT_EXCEL

zones_df, checklist_df = safe_read_excel(excel_path)
zones_df = ensure_label_columns(zones_df)

# Load image
if map_up is not None:
    map_path = Path("uploaded_map.png")
    map_path.write_bytes(map_up.getvalue())
else:
    map_path = DEFAULT_MAP

if map_path.exists():
    base_img = Image.open(map_path).convert("RGBA")
else:
    base_img = Image.new("RGBA", (1400, 900), (245, 245, 245, 255))

# Load polygons
if poly_up is not None:
    polygons = json.loads(poly_up.getvalue().decode("utf-8"))
else:
    polygons = load_polygons(DEFAULT_POLYGONS)

# Session state init
if "polygons" not in st.session_state:
    st.session_state["polygons"] = polygons
if "zones_edit" not in st.session_state:
    st.session_state["zones_edit"] = zones_df

polygons = st.session_state["polygons"]
zones_df = st.session_state["zones_edit"]

# Floor selection
floors = sorted([f for f in zones_df["floor"].unique().tolist() if int(f) >= 0])
selected_floor = st.sidebar.selectbox("층 선택", floors, index=0 if floors else 0)
scenario = st.sidebar.selectbox("시나리오", ["기본(정전)", "쉬는시간 정전(혼잡)", "전교 대피 정전(방송 불안정)"])

zones_df["scenario"] = scenario
floor_zones = zones_df[zones_df["floor"] == selected_floor].copy()

tabs = st.tabs(["1) 지도/구역 설정", "2) 위험지도 보기", "3) 근거(설명) 보기", "4) 체크리스트/인쇄"])

# -------------------------
# Tab 1: Map & Zones
# -------------------------
with tabs[0]:
    st.subheader("1) 지도 업로드/불러오기 + 층 선택 + 구역(Zone) 찍기")

    colA, colB = st.columns([1.3, 1])
    with colA:
        st.write("**현재 지도 미리보기**")
        st.image(base_img, use_container_width=True)

        if not HAS_CANVAS:
            st.warning("그리기 기능(캔버스)을 쓰려면 `streamlit-drawable-canvas`가 필요합니다. requirements.txt 설치 후 실행하세요.")
        else:
            st.markdown("### 구역 사각형 찍기(빠른 MVP)")
            zone_pick = st.selectbox("구역 선택(찍을 대상)", floor_zones["zone_id"].tolist(),
                                     format_func=lambda zid: f"{zid} — {floor_zones.set_index('zone_id').loc[zid, 'display_name']}")
            drawing_mode = st.radio("그리기 모드", ["rect"], horizontal=True)
            st.caption("팁: 교실은 박스 1개로, 복도는 길게 1~3구간만 나눠도 됩니다.")

            canvas_res = st_canvas(
                fill_color="rgba(255, 0, 0, 0.15)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=base_img,
                update_streamlit=True,
                height=min(950, base_img.size[1]),
                width=min(1400, base_img.size[0]),
                drawing_mode=drawing_mode,
                key=f"canvas_{selected_floor}",
            )

            if canvas_res.json_data is not None and len(canvas_res.json_data.get("objects", [])) > 0:
                # Use the last drawn object
                obj = canvas_res.json_data["objects"][-1]
                if obj.get("type") == "rect":
                    poly = rect_to_poly(obj)
                    polygons[zone_pick] = {"poly": poly, "source": "rect"}
                    st.success(f"{zone_pick} 구역 저장 완료! (사각형)")
                    st.session_state["polygons"] = polygons

            # Save buttons
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("폴리곤 저장(로컬 파일)"):
                    save_polygons(DEFAULT_POLYGONS, polygons)
                    st.toast("polygons.json 저장 완료", icon="✅")
            with c2:
                st.download_button(
                    "폴리곤 JSON 다운로드",
                    data=json.dumps(polygons, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="polygons.json",
                    mime="application/json",
                )
            with c3:
                st.metric("이 층 완료 구역", f"{sum(1 for z in floor_zones['zone_id'] if z in polygons)}/{len(floor_zones)}")

    with colB:
        st.markdown("### 구역 테이블(라벨/입력값도 같이 관리)")
        editable_cols = ["zone_id", "display_name", "zone_type", "floor", "risk_label",
                         "survey_darkness", "survey_confusion", "survey_confidence", "sim_congestion",
                         "map_hint", "notes"]
        shown = zones_df[editable_cols].copy()
        edited = st.data_editor(
            shown,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "risk_label": st.column_config.NumberColumn("risk_label (0~100)", min_value=0, max_value=100, step=1),
                "survey_darkness": st.column_config.NumberColumn("설문:어두움(1~5)", min_value=1, max_value=5, step=1),
                "survey_confusion": st.column_config.NumberColumn("설문:혼란(1~5)", min_value=1, max_value=5, step=1),
                "survey_confidence": st.column_config.NumberColumn("설문:길찾기 자신감(1~5)", min_value=1, max_value=5, step=1),
                "sim_congestion": st.column_config.NumberColumn("시뮬:혼잡(0~1)", min_value=0.0, max_value=1.0, step=0.01),
            },
        )
        # Persist edits
        st.session_state["zones_edit"] = edited.merge(zones_df.drop(columns=editable_cols), left_on="zone_id", right_on="zone_id", how="left")

        st.download_button(
            "구역 데이터 CSV 다운로드",
            data=st.session_state["zones_edit"].to_csv(index=False).encode("utf-8-sig"),
            file_name="zones_with_labels.csv",
            mime="text/csv",
        )

# -------------------------
# Tab 2: Risk Map
# -------------------------
with tabs[1]:
    st.subheader("2) 위험 표시(히트맵/구역 색칠)")

    feat_df = build_feature_table(zones_df, polygons)

    model = train_models(feat_df)
    if model is None:
        st.info("모델 학습을 위해서는 **risk_label(0~100)** 이 최소 10개 이상 필요합니다. (설문 평균이나 체크리스트 기반 점수로 채워 넣으세요)")
        st.write("현재 라벨 수:", int(feat_df["risk_label"].notna().sum()))
        # Still show map with labeled zones only
        temp = zones_df.copy()
        temp["pred_score"] = temp["risk_label"]
        img2 = draw_overlay(base_img, temp[temp["floor"] == selected_floor], polygons, "pred_score")
        st.image(img2, use_container_width=True)
    else:
        st.success(f"모델 학습 완료! (교차검증 MAE ≈ {model.mae_cv:.1f})")
        # Predict for all zones
        X_all = feat_df.drop(columns=["risk_label"])
        preds = model.rf.predict(X_all)
        zones_df = zones_df.copy()
        zones_df["pred_score"] = preds
        st.session_state["zones_edit"] = zones_df  # persist

        floor_pred = zones_df[zones_df["floor"] == selected_floor].copy()
        img2 = draw_overlay(base_img, floor_pred, polygons, "pred_score")

        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.image(img2, use_container_width=True)
        with c2:
            st.markdown("### 위험 Top 10")
            top = floor_pred.dropna(subset=["pred_score"]).sort_values("pred_score", ascending=False).head(10)
            st.dataframe(top[["zone_id", "display_name", "zone_type", "pred_score"]], use_container_width=True, hide_index=True)

            st.markdown("### 등급(안전/주의/위험)")
            if top["pred_score"].notna().any():
                q1, q2 = np.quantile(floor_pred["pred_score"].dropna(), [0.6, 0.85])
                def cls(v):
                    if v >= q2: return "위험"
                    if v >= q1: return "주의"
                    return "안전"
                floor_pred["risk_class"] = floor_pred["pred_score"].apply(cls)
                st.dataframe(floor_pred[["zone_id", "display_name", "risk_class", "pred_score"]].sort_values("pred_score", ascending=False).head(15),
                             use_container_width=True, hide_index=True)

# -------------------------
# Tab 3: Explanation
# -------------------------
with tabs[2]:
    st.subheader("3) 구역 클릭/선택 → 근거 설명(설명가능)")

    feat_df = build_feature_table(st.session_state["zones_edit"], polygons)
    model = train_models(feat_df)

    pick = st.selectbox("근거를 볼 구역 선택", floor_zones["zone_id"].tolist(),
                        format_func=lambda zid: f"{zid} — {floor_zones.set_index('zone_id').loc[zid, 'display_name']}")

    row = feat_df[feat_df["zone_id"] == pick].iloc[0]
    zrow = st.session_state["zones_edit"].set_index("zone_id").loc[pick]

    # Display summary
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### 구역 요약")
        st.write({
            "zone_id": pick,
            "이름": str(zrow["display_name"]),
            "타입": str(zrow["zone_type"]),
            "층": int(zrow["floor"]),
            "risk_label(있다면)": None if pd.isna(zrow.get("risk_label", np.nan)) else float(zrow["risk_label"]),
            "pred_score(있다면)": None if pd.isna(zrow.get("pred_score", np.nan)) else float(zrow["pred_score"]),
        })

        st.markdown("### 입력 데이터(근거용)")
        evidence = {
            "설문-어두움(1~5)": None if pd.isna(zrow.get("survey_darkness", np.nan)) else float(zrow["survey_darkness"]),
            "설문-혼란(1~5)": None if pd.isna(zrow.get("survey_confusion", np.nan)) else float(zrow["survey_confusion"]),
            "설문-길찾기 자신감(1~5)": None if pd.isna(zrow.get("survey_confidence", np.nan)) else float(zrow["survey_confidence"]),
            "시뮬-혼잡(0~1)": None if pd.isna(zrow.get("sim_congestion", np.nan)) else float(zrow["sim_congestion"]),
        }
        st.json(evidence, expanded=False)

    with c2:
        st.markdown("### 지도에서 위치 하이라이트")
        # Create a highlight image
        temp = st.session_state["zones_edit"].copy()
        temp["tmp_score"] = 0
        hi = draw_overlay(base_img, temp[temp["floor"] == selected_floor], polygons, "tmp_score")  # just outlines
        hi = hi.convert("RGBA")
        draw = ImageDraw.Draw(hi, "RGBA")
        if pick in polygons and polygons[pick].get("poly"):
            pts = [(float(x), float(y)) for x, y in polygons[pick]["poly"]]
            draw.polygon(pts, outline=(0, 255, 255, 220), width=4)
        st.image(hi, use_container_width=True)

    st.markdown("### 왜 위험한가? (설명 문구 자동 생성)")

    # Template explanations using available evidence
    def sentence_templates(z: pd.Series) -> List[str]:
        parts = []
        # Use filled inputs as "data grounds"
        if not pd.isna(z.get("survey_darkness", np.nan)) and z["survey_darkness"] >= 4:
            parts.append("설문에서 ‘정전 시 어두움’ 점수가 높았습니다.")
        if not pd.isna(z.get("survey_confusion", np.nan)) and z["survey_confusion"] >= 4:
            parts.append("설문에서 ‘혼란/우왕좌왕’ 가능성이 높게 나타났습니다.")
        if not pd.isna(z.get("survey_confidence", np.nan)) and z["survey_confidence"] <= 2:
            parts.append("설문에서 ‘비상구 방향 자신감’이 낮아 길찾기 실패 위험이 있습니다.")
        if not pd.isna(z.get("sim_congestion", np.nan)) and z["sim_congestion"] >= 0.6:
            parts.append("시뮬레이션에서 혼잡도가 높은 구간으로 예측되었습니다.")
        if len(parts) == 0:
            parts.append("현재 입력 데이터가 부족해, 위치/유형 기반으로 보수적으로 예측했습니다.")
        score = zrow.get("pred_score", np.nan)
        if not pd.isna(score):
            prefix = f"**예측 위험 점수 {score:.0f}/100**: "
        else:
            prefix = "**위험 근거**: "
        s1 = prefix + " ".join(parts[:2])
        s2 = "근거 데이터: " + ", ".join([k for k,v in evidence.items() if v is not None]) + " (입력된 항목 기준)"
        return [s1, s2]

    for s in sentence_templates(zrow):
        st.write("• " + s)

    if model is not None:
        st.markdown("### 모델 기반 설명(상위 기여 요인)")
        contrib = explain_zone(model, row)
        st.dataframe(pd.DataFrame(contrib, columns=["요인", "기여(+) 위험↑ / (-) 위험↓"]).head(6), use_container_width=True, hide_index=True)
        st.caption("※ 기여도는 단순 선형모델(Ridge) 기준이며, 실제 예측은 RandomForest 결과를 사용합니다.")
    else:
        st.info("모델 설명을 보려면 risk_label이 최소 10개 이상 필요합니다.")

# -------------------------
# Tab 4: Checklist & Print
# -------------------------
with tabs[3]:
    st.subheader("4) 인쇄 가능한 체크리스트 (담임/행정실용)")

    st.markdown("### 체크리스트(15개) 확인/수정")
    checklist_edit = st.data_editor(checklist_df, use_container_width=True, hide_index=True, num_rows="fixed")
    st.download_button(
        "체크리스트 CSV 다운로드",
        data=checklist_edit.to_csv(index=False).encode("utf-8-sig"),
        file_name="checklist.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown("### 인쇄용 PDF 생성")
    st.caption("PDF에는 (1) 층/시나리오, (2) 위험 Top 구역, (3) 체크리스트 항목(체크박스)이 들어갑니다.")

    zones_now = st.session_state["zones_edit"].copy()
    floor_now = zones_now[zones_now["floor"] == selected_floor].copy()

    # Determine risk values for printing
    score_col = "pred_score" if "pred_score" in floor_now.columns and floor_now["pred_score"].notna().any() else "risk_label"
    topk = floor_now.dropna(subset=[score_col]).sort_values(score_col, ascending=False).head(10)

    def build_pdf_bytes() -> bytes:
        from io import BytesIO
        buf = BytesIO()
        c = pdf_canvas.Canvas(buf, pagesize=A4)
        w, h = A4

        x0 = 18 * mm
        y = h - 18 * mm

        def line(txt, dy=6.5*mm, size=11, bold=False):
            nonlocal y
            c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
            c.drawString(x0, y, txt)
            y -= dy

        line("AI 학교 위험지도 체크리스트 (인쇄용)", size=16, bold=True, dy=9*mm)
        line(f"- 층: {selected_floor}F", bold=False)
        line(f"- 시나리오: {scenario}", bold=False)
        line(f"- 점수 기준: {score_col}", bold=False)
        y -= 3*mm

        line("1) 위험 구역 Top 10", bold=True, dy=8*mm)
        if len(topk) == 0:
            line("  (아직 라벨/예측 점수가 없습니다. risk_label을 입력하세요.)", size=10)
        else:
            for _, r in topk.iterrows():
                line(f"□ {r['zone_id']}  {r['display_name']}  ({float(r[score_col]):.0f}/100)", size=10, dy=6*mm)

        y -= 2*mm
        line("2) 점검 체크리스트(15)", bold=True, dy=8*mm)
        for _, r in checklist_edit.iterrows():
            item = str(r.get("item", "")).strip()
            if not item:
                continue
            # wrap
            text = f"□ {item}"
            # naive wrapping
            max_chars = 55
            if len(text) <= max_chars:
                line(text, size=10, dy=6*mm)
            else:
                line(text[:max_chars], size=10, dy=6*mm)
                line("   " + text[max_chars:], size=10, dy=6*mm)

            if y < 25*mm:
                c.showPage()
                y = h - 18*mm

        y -= 4*mm
        line("메모:", bold=True, dy=8*mm)
        for _ in range(6):
            line("________________________________________________________________________", size=10, dy=7*mm)

        c.showPage()
        c.save()
        return buf.getvalue()

    pdf_bytes = build_pdf_bytes()
    st.download_button(
        "인쇄용 체크리스트 PDF 다운로드",
        data=pdf_bytes,
        file_name=f"체크리스트_{selected_floor}F.pdf",
        mime="application/pdf",
    )

    st.markdown("#### 설명 문구 예시(인쇄물/보고서에 그대로 사용 가능)")
    st.write("• “본 체크리스트는 설문·시뮬 기반 위험 예측 결과를 바탕으로 우선 점검 구역을 제시합니다. (시나리오별로 결과가 달라질 수 있음)”")
    st.write("• “점수/등급은 학교 내부 데이터로 학습한 모델의 예측이며, 불확실 구역은 현장 확인 후 조치 여부를 결정합니다.”")

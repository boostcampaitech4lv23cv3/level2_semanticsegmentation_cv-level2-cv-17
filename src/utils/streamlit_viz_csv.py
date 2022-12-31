import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pycocotools.coco import COCO

CSV_DIR = "/opt/ml/input/code/submission/"


@st.cache
def load_data():
    """데이터(이미지) 정보, 컬러 팔레트, 클래스명을 캐시"""
    coco = COCO("/opt/ml/input/data/test.json")
    image_names = [coco.loadImgs(id)[0]["file_name"] for id in coco.getImgIds()]

    image_names.append(None)

    palette = pd.read_csv("/opt/ml/input/code/class_dict.csv")
    color_palette = [
        (r, g, b) for r, g, b in zip(palette["r"], palette["g"], palette["b"])
    ]

    category_names = [
        "Backgroud",
        "General trash",
        "Paper",
        "Paper pack",
        "Metal",
        "Glass",
        "Plastic",
        "Styrofoam",
        "Plastic bag",
        "Battery",
        "Clothing",
    ]

    return image_names, color_palette, category_names, len(image_names)


def load_csv_list():
    """서버에 저장된 csv 파일의 리스트를 업데이트"""
    file_names = []
    dir_list = os.listdir(CSV_DIR)
    for file_name in dir_list:
        splitted = file_name.split(".")
        if len(splitted) == 2 and splitted[1].lower() == "csv":
            file_names.append(file_name)

    file_names.append(None)

    return file_names, len(file_names)


def set_session_state():
    """Streamlit 세션 정보 초기 세팅"""
    global N_IMAGES, N_CSV

    if "img_idx" not in st.session_state:
        st.session_state.img_idx = N_IMAGES - 1

    if "show_img" not in st.session_state:
        st.session_state.show_img = False

    if "csv_name_list" not in st.session_state:
        st.session_state.csv_name_list = load_csv_list()[0]

    if "csv_idx" not in st.session_state:
        st.session_state.csv_idx = N_CSV - 1

    if "viz_csv" not in st.session_state:
        st.session_state.viz_csv = False

    if "file_upldd" not in st.session_state:
        st.session_state.file_upldd = False


def set_image_session_states():
    """시각화할 사진의 인덱스를 결정"""
    for state in ("sbox_img_idx", "prev_img_btn", "next_img_btn"):
        if state not in st.session_state:
            return

    increment_idx = lambda idx, inc, max: (idx + inc + max) % max
    show_img = lambda idx: not idx == N_IMAGES - 1

    if st.session_state.prev_img_btn:
        st.session_state.img_idx = increment_idx(st.session_state.img_idx, -1, N_IMAGES)
    elif st.session_state.next_img_btn:
        st.session_state.img_idx = increment_idx(st.session_state.img_idx, 1, N_IMAGES)
    else:
        st.session_state.img_idx = st.session_state.sbox_img_idx
    st.session_state.show_img = show_img(st.session_state.img_idx)


def set_csv_session_states():
    """리뷰할 csv 파일과 관련 세션정보를 설정"""
    if "sbox_csv_name" not in st.session_state:
        return

    if st.session_state.file_upldd:
        st.session_state.csv_name_list = load_csv_list()[0]
        st.session_state.csv_idx = st.session_state.csv_name_list.index(
            st.session_state.upldd_file.name
        )
        st.session_state.viz_csv = True
        st.session_state.file_upldd = False
    else:
        if st.session_state.sbox_csv_name is None:
            st.session_state.viz_csv = False
        else:
            st.session_state.csv_idx = st.session_state.csv_name_list.index(
                st.session_state.sbox_csv_name
            )
            st.session_state.viz_csv = True


def save_csv(section):
    """업로드된 csv 파일을 서버에 저장"""
    if st.session_state.upldd_file is None:
        return

    path = os.path.join(CSV_DIR, st.session_state.upldd_file.name)

    if os.path.exists(path):
        section.error(
            f"\
                `{st.session_state.upldd_file.name}`\
                already exists in `{CSV_DIR}`.\
                Please change the name of the file to save.\
            "
        )

    else:
        with open(path, "wb") as f:
            f.write(st.session_state.upldd_file.getbuffer())

    st.session_state.file_upldd = True


def render_csv_uploader(section):
    """파일 업로더"""
    section.file_uploader(
        "Upload File", type=".csv", key="upldd_file", on_change=save_csv, args=[section]
    )


def render_csv_selectbox(section):
    """csv 파일 선택박스"""
    global csv_names, n_files

    section.selectbox(
        "Select `CSV` File",
        st.session_state.csv_name_list,
        index=st.session_state.csv_idx,
        key="sbox_csv_name",
    )


def render_image_buttons(section):
    """이전, 다음 이미지 버튼"""
    left_col, right_col = section.columns(2)

    left_col.button("Previous Image", key="prev_img_btn")

    right_col.button("Next Image", key="next_img_btn")


def render_image_selectbox(section):
    """이미지 선택박스"""
    global IMG_NAMES, N_IMAGES

    section.selectbox(
        "Select Image",
        range(N_IMAGES),
        index=st.session_state.img_idx,
        format_func=lambda idx: IMG_NAMES[idx],
        key="sbox_img_idx",
    )

    btn_cont = section.container()
    render_image_buttons(btn_cont)


def render_color_map(section):
    """색상 대조표"""
    global COLOR_PALETTE, CAT_NAMES

    rgb2hex = lambda rgb: "#%02x" % rgb[0] + "%02x" % rgb[1] + "%02x" % rgb[2]

    for cat_name, color in zip(CAT_NAMES, COLOR_PALETTE):
        section.markdown(
            f"""<p style='
                background-color:{rgb2hex(color)};
                color:#FFFFFF;
                font-size: 30px;
                font-weight: 700'>{cat_name}</p>""",
            unsafe_allow_html=True,
        )


def load_image():
    """이미지 로드"""
    image_name = IMG_NAMES[st.session_state.img_idx]
    image = cv2.imread(os.path.join("/opt/ml/input/data", image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    return image


def generate_mask(section, csv):
    """마스크 생성"""
    img_idx = st.session_state.img_idx
    image_name = IMG_NAMES[st.session_state.img_idx]
    csv_idx = st.session_state.csv_idx
    csv_name = st.session_state.csv_name_list[csv_idx]

    mask = np.zeros((256, 256, 3))
    info = csv.loc[csv["image_id"] == image_name]
    if len(info) == 0:
        section.error(
            f"\
                Cannot find `{image_name}`\
                from `{csv_name}`!\
                Mask has not been generated!"
        )
    else:
        pixels = info["PredictionString"][img_idx].split()
        for i, category in enumerate(pixels):
            mask[i // 256, i % 256, :] = COLOR_PALETTE[int(category)]
    mask = cv2.resize(mask, (512, 512), cv2.INTER_LINEAR)
    mask /= 255.0

    return mask


def overwrite_mask(image, mask):
    """마스킹 이미지 생성"""
    if "opacity" not in st.session_state:
        opacity = 0.65
    else:
        opacity = st.session_state.opacity

    return image * (1 - opacity) + mask * opacity


def generate_images(section):
    """시각화할 이미지들 생성"""
    global COLOR_PALETTE, N_IMAGES

    csv_name = st.session_state.csv_name_list[st.session_state.csv_idx]
    image_name = IMG_NAMES[st.session_state.img_idx]
    csv = pd.read_csv(os.path.join(CSV_DIR, csv_name))

    left, right = section.columns(2)
    left.success(f"File: `{csv_name}`")
    right.success(f"Image: `{image_name}`")

    image = load_image()
    mask = generate_mask(section, csv)
    masked = overwrite_mask(image, mask)

    return image, masked, mask


def render_images(section):
    """이미지를 화면에 띄움"""
    if not st.session_state.show_img and not st.session_state.viz_csv:
        section.warning("Please select a file to review.")
        section.warning("Please select an image to view.")
        return
    elif not st.session_state.show_img:
        section.warning("Please select an image to view.")
        return
    elif not st.session_state.viz_csv:
        section.warning("File has not been selected. Please upload or select a file.")
        return

    original, masked, mask = generate_images(section)

    cols = section.columns(3)

    cols[0].subheader("Original Image")
    cols[0].image(original)

    cols[1].subheader("Mask Overwritten Image")
    cols[1].image(masked)
    cols[1].slider(
        "Mask Opacity",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
        key="opacity",
    )

    cols[2].subheader("Output Mask")
    cols[2].image(mask)


def render_control_panel(section):
    """제어 섹션"""
    section.title("Select `CSV` File and Image")

    csv_sbox_cont = section.container()
    render_csv_selectbox(csv_sbox_cont)

    img_sbox_cont = section.container()
    render_image_selectbox(img_sbox_cont)

    cmap_expander = section.expander("View color map")
    render_color_map(cmap_expander)

    uploade_expander = section.expander("Click here to upload your file")
    render_csv_uploader(uploade_expander)


def render_result(section):
    """결과"""
    section.title("Output File Visualizer")

    render_images(section)


# Initialize
st.set_page_config(layout="wide", page_title="Output File Visualizer")
# Load prerequisite data
(
    IMG_NAMES,
    COLOR_PALETTE,
    CAT_NAMES,
    N_IMAGES,
) = load_data()

N_CSV = load_csv_list()[1]

# Set session states
set_session_state()

# Decide image index
set_image_session_states()

# Decide CSV file
set_csv_session_states()

# Render page
render_control_panel(st.sidebar)
render_result(st)

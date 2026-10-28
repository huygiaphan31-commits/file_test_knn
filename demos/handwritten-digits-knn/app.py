"""Giao diện Streamlit cho demo nhận diện chữ viết tay bằng KNN tự cài."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from config import (
    DATA_DIR,
    DEFAULT_CLASS_IDS,
    FIXED_EXPERIMENT_K,
    RANDOM_STATE,
    WEB_MAX_TRAIN_PER_CLASS,
)
from data_loader import load_emnist_training
from knn import ManualKNN
from line_recognition import crop_line_regions, recognize_line
from main import fit_classifier
from preprocessing import load_grayscale_image, preprocess_character_image_array

SINGLE_CHARACTER_MODE = "Một ký tự"
SINGLE_LINE_MODE = "Một dòng chữ"
WEB_K_SELECTION_MODE = "fixed_101"
WEB_CONFIGURATION_KEY = (WEB_MAX_TRAIN_PER_CLASS, WEB_K_SELECTION_MODE)


@dataclass(frozen=True)
class PredictionView:
    text: str
    processed_image: np.ndarray | None
    annotated_image: np.ndarray | None
    crops: tuple[np.ndarray, ...]
    region_predictions: tuple[str, ...]


@dataclass(frozen=True)
class LoadedDemoModel:
    model: ManualKNN
    selected_k: int
    mapping: dict[int, str]
    training_count: int
    validation_count: int
    validation_macro_f1: float
    max_train_per_class: int
    k_selection_mode: str

    @property
    def configuration_key(self) -> tuple[int, str]:
        return self.max_train_per_class, self.k_selection_mode


def model_matches_configuration(
    model_info: LoadedDemoModel | None,
    configuration_key: tuple[int, str],
) -> bool:
    return (
        model_info is not None
        and model_info.configuration_key == configuration_key
        and model_info.selected_k == FIXED_EXPERIMENT_K
    )


def prediction_key(
    image_bytes: bytes,
    mode: str,
    configuration_key: tuple[int, str],
) -> tuple[str, str, tuple[int, str]]:
    content_digest = hashlib.sha256(image_bytes).hexdigest()
    return content_digest, mode, configuration_key


@st.cache_resource(show_spinner=False)
def _model_cache_state() -> dict[str, int]:
    return {"generation": 0}


@st.cache_resource(show_spinner=False, max_entries=1)
def load_demo_model(
    data_dir: str,
    max_train_per_class: int = WEB_MAX_TRAIN_PER_CLASS,
    k_selection_mode: str = WEB_K_SELECTION_MODE,
) -> LoadedDemoModel:
    """Nạp cấu hình web duy nhất: 500 ảnh/lớp và k=101."""
    if (max_train_per_class, k_selection_mode) != WEB_CONFIGURATION_KEY:
        raise ValueError(
            "Web chỉ hỗ trợ 500 ảnh/lớp với chế độ k cố định bằng 101."
        )

    mapping, training_data = load_emnist_training(
        Path(data_dir),
        DEFAULT_CLASS_IDS,
        max_train_per_class,
        RANDOM_STATE,
    )
    (
        model,
        selected_k,
        validation_scores,
        _,
        validation_indices,
        _,
    ) = fit_classifier(
        training_data,
        DEFAULT_CLASS_IDS,
        mapping,
        k_candidates=(FIXED_EXPERIMENT_K,),
    )
    if FIXED_EXPERIMENT_K not in validation_scores:
        raise ValueError(
            f"Không thể xác nhận k={FIXED_EXPERIMENT_K} trên validation hiện tại."
        )
    selected_k = FIXED_EXPERIMENT_K

    return LoadedDemoModel(
        model=model,
        selected_k=selected_k,
        mapping=mapping,
        training_count=len(training_data.labels),
        validation_count=len(validation_indices),
        validation_macro_f1=validation_scores[selected_k],
        max_train_per_class=max_train_per_class,
        k_selection_mode=k_selection_mode,
    )


def refresh_demo_model(
    data_dir: str,
    max_train_per_class: int = WEB_MAX_TRAIN_PER_CLASS,
    k_selection_mode: str = WEB_K_SELECTION_MODE,
) -> LoadedDemoModel:
    """Invalidate the shared model cache and synchronously load fresh train data."""
    _model_cache_state()["generation"] += 1
    load_demo_model.clear()
    return load_demo_model(data_dir, max_train_per_class, k_selection_mode)


def predict_uploaded_image(
    image_bytes: bytes,
    mode: str,
    model: ManualKNN,
    selected_k: int,
    mapping: dict[int, str],
) -> PredictionView:
    """Chạy ảnh tải lên qua preprocessing và classifier hiện có."""
    if mode not in (SINGLE_CHARACTER_MODE, SINGLE_LINE_MODE):
        raise ValueError(f"Chế độ ảnh không hợp lệ: {mode}")

    with tempfile.TemporaryDirectory() as temporary:
        image_path = Path(temporary) / "uploaded-image.png"
        image_path.write_bytes(image_bytes)
        grayscale = load_grayscale_image(image_path)

        if mode == SINGLE_CHARACTER_MODE:
            processed_image, vector = preprocess_character_image_array(grayscale)
            class_id, _ = model.predict_one(vector, selected_k)
            if class_id not in mapping:
                raise ValueError(f"Không có ký tự cho class ID dự đoán {class_id}.")
            return PredictionView(
                text=mapping[class_id],
                processed_image=processed_image,
                annotated_image=None,
                crops=(),
                region_predictions=(),
            )

        result = recognize_line(image_path, model, selected_k, mapping)
        annotated = Image.fromarray(grayscale, mode="L").convert("RGB")
        draw = ImageDraw.Draw(annotated)
        for index, box in enumerate(result.boxes, start=1):
            draw.rectangle(box, outline=(220, 30, 30), width=2)
            draw.text((box[0], max(0, box[1] - 14)), str(index), fill=(220, 30, 30))

        return PredictionView(
            text=result.text,
            processed_image=None,
            annotated_image=np.asarray(annotated),
            crops=crop_line_regions(grayscale, result.boxes),
            region_predictions=result.region_predictions,
        )


def _render_pipeline_explanation() -> None:
    with st.expander("Các bước nhận diện"):
        st.markdown(
            f"""
1. **Tiền xử lý:** chuyển ảnh sang grayscale và chuẩn hóa glyph bằng pipeline dùng chung với EMNIST.
2. **Phân đoạn (chỉ chế độ dòng):** tìm vùng ký tự theo connected components và sắp xếp trái sang phải.
3. **Khoảng cách:** so sánh vector pixel 784 chiều với các mẫu train bằng khoảng cách Euclid.
4. **Chọn láng giềng:** giao diện dùng cố định `k={FIXED_EXPERIMENT_K}`.
5. **Bỏ phiếu:** chọn nhãn có nhiều láng giềng nhất; ghép nhãn theo thứ tự đọc.

KNN được tự cài trong `knn.py`; giao diện không dùng CNN hoặc OCR.
"""
        )


def main() -> None:
    st.set_page_config(page_title="Nhận diện chữ viết tay KNN", page_icon="✍️")
    st.title("Nhận diện chữ viết tay bằng KNN")
    st.caption("EMNIST ByClass · 62 lớp · thuật toán KNN tự cài")
    st.info(
        "Mô hình chỉ được nạp khi bạn bấm nút bên dưới. Ứng dụng không chạy "
        "đánh giá test và không tải dataset tự động."
    )

    with st.sidebar:
        st.subheader("Chuẩn bị mô hình")
        max_train_per_class, k_selection_mode = WEB_CONFIGURATION_KEY
        configuration_key = WEB_CONFIGURATION_KEY
        cache_generation = _model_cache_state()["generation"]
        stored_model = st.session_state.get("demo_model")
        stored_generation = st.session_state.get("demo_model_generation")
        if stored_model is not None and (
            stored_generation != cache_generation
            or not model_matches_configuration(stored_model, configuration_key)
        ):
            st.session_state.pop("demo_model", None)
            st.session_state.pop("last_prediction", None)
            st.session_state.pop("demo_model_generation", None)
        st.write(
            f"Cấu hình duy nhất: 500 ảnh/lớp (tối đa 31.000 mẫu), "
            f"k={FIXED_EXPERIMENT_K} cố định."
        )
        load_column, refresh_column = st.columns(2)
        load_requested = load_column.button(
            "Nạp model",
            use_container_width=True,
            help="Dùng model đã cache nếu cấu hình hiện tại đã được nạp.",
        )
        refresh_requested = refresh_column.button(
            "Làm mới model",
            use_container_width=True,
            help="Xóa cache model dùng chung, đọc lại train và tạo KNN mới.",
        )
        if load_requested or refresh_requested:
            if refresh_requested:
                st.session_state.pop("demo_model", None)
                st.session_state.pop("demo_model_generation", None)
                st.session_state.pop("last_prediction", None)
            try:
                status = (
                    "Đang xóa cache model, đọc lại train và chuẩn bị KNN..."
                    if refresh_requested
                    else "Đang đọc train và chuẩn bị KNN..."
                )
                with st.spinner(status):
                    model_info = (
                        refresh_demo_model(
                            str(DATA_DIR),
                            max_train_per_class,
                            k_selection_mode,
                        )
                        if refresh_requested
                        else load_demo_model(
                            str(DATA_DIR),
                            max_train_per_class,
                            k_selection_mode,
                        )
                    )
                    st.session_state["demo_model"] = model_info
                    st.session_state["demo_model_generation"] = (
                        _model_cache_state()["generation"]
                    )
                    st.session_state.pop("last_prediction", None)
            except (FileNotFoundError, OSError, ValueError) as error:
                st.session_state.pop("demo_model", None)
                st.session_state.pop("demo_model_generation", None)
                st.session_state.pop("last_prediction", None)
                st.error(f"Không thể chuẩn bị mô hình: {error}")

        stored_model = st.session_state.get("demo_model")
        model_info = (
            stored_model
            if (
                st.session_state.get("demo_model_generation")
                == _model_cache_state()["generation"]
                and model_matches_configuration(stored_model, configuration_key)
            )
            else None
        )
        if model_info is not None:
            st.success("Mô hình đã sẵn sàng.")
            st.metric("Mẫu train đã nạp", f"{model_info.training_count:,}")
            st.metric("k sử dụng", model_info.selected_k)
            st.metric("Validation macro-F1", f"{model_info.validation_macro_f1:.4f}")
            st.caption(
                f"Validation gồm {model_info.validation_count:,} mẫu. "
                "Đây không phải test accuracy."
            )
        else:
            st.info("Hãy nạp mô hình cho cấu hình đang chọn trước khi nhận diện.")

    mode = st.radio(
        "Chọn chế độ",
        (SINGLE_CHARACTER_MODE, SINGLE_LINE_MODE),
        horizontal=True,
    )
    uploaded = st.file_uploader(
        "Chọn ảnh PNG hoặc JPG",
        type=("png", "jpg", "jpeg"),
        accept_multiple_files=False,
    )
    uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
    current_prediction_key = (
        prediction_key(uploaded_bytes, mode, configuration_key)
        if uploaded_bytes is not None
        else None
    )
    previous_prediction = st.session_state.get("last_prediction")
    if (
        previous_prediction is not None
        and previous_prediction[0] != current_prediction_key
    ):
        st.session_state.pop("last_prediction", None)

    if uploaded is not None:
        st.image(uploaded_bytes, caption=uploaded.name, use_container_width=True)

    stored_model = st.session_state.get("demo_model")
    model_info = (
        stored_model
        if model_matches_configuration(stored_model, configuration_key)
        else None
    )
    if st.button(
        "Nhận diện",
        type="primary",
        disabled=uploaded is None or model_info is None,
        use_container_width=True,
    ):
        try:
            with st.spinner("Đang tiền xử lý và dự đoán..."):
                prediction = predict_uploaded_image(
                    uploaded_bytes,
                    mode,
                    model_info.model,
                    model_info.selected_k,
                    model_info.mapping,
                )
            st.session_state["last_prediction"] = (
                current_prediction_key,
                prediction,
            )
        except (OSError, ValueError) as error:
            st.error(f"Không thể nhận diện ảnh: {error}")

    last_prediction = st.session_state.get("last_prediction")
    if (
        current_prediction_key is not None
        and last_prediction is not None
        and last_prediction[0] == current_prediction_key
    ):
        prediction = last_prediction[1]
        st.subheader("Kết quả")
        st.metric("Dự đoán", prediction.text or "Không tìm thấy ký tự")
        if prediction.processed_image is not None:
            st.image(
                prediction.processed_image,
                caption="Glyph sau chuẩn hóa 28×28",
                width=224,
            )
        if prediction.annotated_image is not None:
            st.image(
                prediction.annotated_image,
                caption="Các vùng phân đoạn, theo thứ tự từ trái sang phải",
                use_container_width=True,
            )
            if prediction.crops:
                st.write("Crop và nhãn dự đoán từng vùng:")
                columns = st.columns(min(5, len(prediction.crops)))
                for index, (crop, label) in enumerate(
                    zip(prediction.crops, prediction.region_predictions), start=1
                ):
                    with columns[(index - 1) % len(columns)]:
                        st.image(crop, caption=f"Vùng {index}: {label}")

    _render_pipeline_explanation()
    st.caption(
        "Kết quả ảnh ngoài là dự đoán, không phải nhãn đúng. "
        "Mô hình chỉ có lớp 0–9, A–Z và a–z; phân đoạn chuỗi là heuristic."
    )


if __name__ == "__main__":
    main()

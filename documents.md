
# TÀI LIỆU KIẾN TRÚC & TRIỂN KHAI HỆ THỐNG MLOps
**Tên dự án:** Hệ thống Nhận diện Bệnh trên Lá cây (Plant Disease Classification)
**Stack Công nghệ:** PyTorch, FastAPI, Airflow, MLflow, MinIO, Docker, GitHub Actions, Google Cloud.
**Stack track and dardboard**: MLflow, Grafana, Prometheus, DVC, promtail
**Stack train**: PyTorch, Optuna, Hydra, Kaggle GPU, MinIO
**Stack demo**: Streamlit


## 1. Cấu trúc mã nguồn (Hybrid Treefile)
```text
plant-disease-mlops/
├── .github/                           # [1] CI/CD & TỰ ĐỘNG HÓA
│   └── workflows/
│       └── deploy-api-cloudrun.yml    # Pipeline tự động build Image và deploy lên Cloud Run
│
├── infra/                             # [2] HẠ TẦNG MLOps (Triển khai trên máy ảo GCE)
│   ├── docker-compose.yml             # Bộ não quản lý: MinIO, MLflow, Airflow, DB, Redis
│   ├── .env.infra.example             # Biến môi trường và mật khẩu hạ tầng
│   └── config_monitoring/             # Cấu hình Monitoring
│       ├── prometheus.yml             # Bộ kéo metrics từ API
│       ├── promtail-config.yml        # Bộ thu thập log hệ thống
│       └── grafana-datasources.yml    # Kết nối data source cho Dashboard
│
├── dags/                              # [3] PIPELINE ĐIỀU PHỐI (Chạy bởi Airflow)
│   ├── utils/
│   │   └── kaggle_hook.py             # Hàm tiện ích kết nối Kaggle API
│   └── train_moco_pipeline.py         # DAG Pipeline: Kéo data -> Đẩy Job lên Kaggle -> Lưu Model
│
├── src/                               # [4] SOURCE CODE LÕI (Ứng dụng & Huấn luyện)
│   ├── api/                           # ---> MICROSERVICE: SERVING (Chỉ chạy khi deploy)
│   │   ├── main.py                    # Ứng dụng FastAPI (Endpoints: /predict, /health, /metrics)
│   │   └── schemas.py                 # Pydantic models (Validate Input/Output)
│   │
│   ├── model/                         # ---> CORE: THƯ VIỆN DÙNG CHUNG
│   │   ├── model.py                   # Cấu trúc mạng Downstream Classification
│   │   └── inference_handler.py       # Xử lý kết nối MinIO, kéo file .pth vào RAM
│   │
│   ├── train_kaggle/                  # ---> JOB: TRAINING (Đẩy lên Kaggle thực thi)
│   │   ├── train.py                   # Code huấn luyện tích hợp Optuna, Hydra và MLflow Tracking
│   │   ├── dataset.py                 # Logic xử lý hình ảnh và Augmentations
│   │   ├── config.yaml                # Cấu hình huấn luyện
│   │   └── requirements_kaggle.txt    # Thư viện tính toán nặng: torch-gpu, optuna, kaggle
│   │
│   └── Dockerfile                     # [!] Bản thiết kế Image cho API (Nằm gọn trong src)
│
├── notebooks/                         # [5] MÔI TRƯỜNG R&D (Local)
│   └── 01_moco_loss_test.ipynb        # Thử nghiệm thuật toán thuần túy
│
├── data/                              # [6] DỮ LIỆU THÔ (Không push lên Git)
│   └── raw/                           # Ảnh mẫu phục vụ test local
│
├── test/                              # pytest cho ci
│   └── test_api.py                    # Test API
│   └── conftest.py                    # pytest config
│
├── .dvc/                              # Cấu hình Data Version Control (Kết nối với MinIO)
├── .dvcignore                         # Loại trừ file rác khỏi quá trình track data
├── .gitignore                         # Loại trừ data/, .env, weights/ khỏi mã nguồn
├── requirements.txt                   # [!] Thư viện cực nhẹ cho API (fastapi, torch-cpu, opencv-headless)
└── README.md                          # Tài liệu dự án
```

---

## 2. Lộ trình Triển khai (Phases) & Cấu hình Cốt lõi

### Phase 0: Khởi tạo Repo & Research (Git Init & Code Chay)
*   **Mục tiêu:** Chứng minh kiến trúc thuật toán hoạt động được trên dữ liệu lá cây và kiểm soát mã nguồn ngay từ ngày đầu.
*   **Khởi tạo Tech stack:** Khởi tạo Git (`git init`), tạo file `.gitignore` (để chặn thư mục `data/` thô, `__pycache__`, file weights `.pth` rác) và đẩy lên một repository trống trên GitHub.
*   **Hành động R&D:** Viết code thuần bằng PyTorch. Tập trung xử lý kiến trúc mô hình, viết data loader. In log loss ra console để xác nhận model hội tụ.
*   **Hành động GitHub:** Commit code thường xuyên với thông điệp rõ ràng (ví dụ: `feat: add initial architecture`). Điều này giúp bạn dễ dàng rollback nếu hướng R&D đi vào ngõ cụt.

### Phase 1: Thêm "Mắt thần" (Tích hợp MLflow & Optuna local)
*   **Mục tiêu:** Theo dõi các experiment, tối ưu tham số tự động và quản lý tính năng mới qua Git Branches.
*   **Khởi tạo Tech stack:** Chạy lệnh `mlflow server` local.
*   **Hành động MLOps:** Import thư viện `mlflow` và bọc hàm train bằng `optuna` để tự dò tham số cốt lõi. Tránh nhúng tham số fine-tune thủ công vào code.
*   **Hành động GitHub:** Tạo một nhánh mới (ví dụ: `git checkout -b feature/experiment-tracking`). Sau khi code chạy ổn định và log được lên MLflow UI, đẩy nhánh này lên GitHub và thực hiện Merge vào nhánh `main`.

### Phase 2: Phân tách Code & Data (MinIO, DVC & GitHub)
*   **Mục tiêu:** Quản lý vòng đời dữ liệu lớn (Versioning) song song với vòng đời mã nguồn.
*   **Khởi tạo Tech stack:** Chạy `docker-compose up -d minio` để dựng kho Object Storage local. Thiết lập `dvc init`.
*   **Hành động DVC:** Chuyển dữ liệu lá cây lên MinIO. DVC sẽ tính toán mã hash và tạo ra các file theo dõi định dạng `.dvc` (chỉ nặng vài KB).
*   **Hành động GitHub:** Không bao giờ push ảnh lá cây lên GitHub. Bạn chỉ commit các file `.dvc` và cấu hình `.dvc/config` lên GitHub. Lúc này, GitHub giữ "bản đồ" trỏ tới dữ liệu, còn MinIO giữ dữ liệu thật.

### Phase 3: Tự động hóa với Orchestration (Airflow)
*   **Mục tiêu:** Đóng gói kịch bản R&D thành luồng chạy tự động và chốt cấu trúc thư mục repo.
*   **Khởi tạo Tech stack:** Bổ sung Airflow (Postgres, Redis) vào cụm hạ tầng local.
*   **Hành động MLOps:** Sử dụng Airflow TaskFlow API (`@task`) để viết DAG bằng Python thuần. Gom các script thành 1 task tải data, 1 task trigger API gọi Kaggle, 1 task kéo model.
*   **Hành động GitHub:** Push toàn bộ cấu trúc thư mục mới (bao gồm `dags/`, `infra/`, `src/`) lên GitHub. Repository của bạn lúc này đã trở thành một hệ thống phần mềm hoàn chỉnh, sẵn sàng cho thành viên khác clone về và chạy.

### Phase 4: Đưa lên Production (Cloud Run, CI/CD GitHub Actions)
*   **Mục tiêu:** Tự động hóa hoàn toàn quy trình đóng gói và triển khai ứng dụng. Đây là nơi GitHub thể hiện quyền năng mạnh nhất.
*   **Khởi tạo Tech stack:** Bọc code inference bằng FastAPI, viết file `Dockerfile` chuẩn trong thư mục `src/` và cấu hình Monitoring (Grafana/Prometheus).
*   **Hành động GitHub Actions:** Cấu hình file workflow `.github/workflows/deploy-api-cloudrun.yml`. Từ thời điểm này, mỗi khi bạn gõ lệnh `git push origin main`, GitHub Actions sẽ tự động kích hoạt luồng CI/CD: Tự động build Docker Image, tự động ném Image đó sang Google Artifact Registry, và tự động cập nhật API mới nhất lên Google Cloud Run.
# **Đồ án môn học IS353 - Mạng Xã Hội** 
# Dự án: Mô hình dự đoán khả năng bị cho thôi học của sinh viên UIT

## Thông tin nhóm

**Tên nhóm:** Nhóm 8  

| Tên thành viên             | MSSV       |
|----------------------------------|------------|
| Trần Phương Lâm          | 21521059   |
| Trương Cao Phúc             | 21521300   |
| Đoàn Ngọc Thanh Sơn      | 21521385   |
| Nguyễn Huy Hoàng           | 21522093   |
| Hồ Nhật Huy               | 21522140   |
| Phạm Quang Hiếu            | 21520235   |
| Nguyễn Thượng Phúc        | 22521134   |

## Mô tả ngắn
Xây dựng một mô hình có khả năng dự đoán với độ tin cậy cao khả năng bị cho thôi học của sinh viên UIT dựa trên dữ liệu do trường UIT cung cấp. Ngoài ra, xây dựng một trang web dự đoán để sinh viên có thể trải nghiệm và từ đó đề xuất kế hoạch khắc phục.

**Link web:** [Demo App](https://demo-app-bf7bwtltn7xslunytmxymy.streamlit.app/)

## Cài đặt

### Bước 1: Clone dự án

```bash
git clone https://github.com/truongcaophuc/demo-app.git
cd demo-app
```

### Bước 2: Cài đặt các thư viện yêu cầu

```bash
pip install -r requirements.txt
```

### Bước 3: Chạy dự án

```bash
streamlit run main.py
```

## Hướng dẫn sử dụng
1. Sau khi chạy lệnh `streamlit run main.py`, mở trình duyệt web và truy cập địa chỉ hiện thị trong terminal.
2. Nhập các thông tin yêu cầu trên giao diện trang web.
3. Nhận kết quả dự đoán và tham khảo những khuyên nghị khắc phục.

## Link video demo
[Video Demo](https://www.youtube.com/watch?v=h20sYwGhbdA)

## Cấu trúc dự án

```
demo-app/
├───cloud                  # Thư mục làm việc với các dịch vụ cloud
├───data                   # Thư mục chứa dữ liệu mẫu
├───demo                   # Thư mục demo các file test
├───docs                   # Tài liệu và hướng dẫn
├───notebooks              # Thư mục chứa notebook jupyter
├───src                    # Thư mục chính chứa code nguồn
├───tests                  # Thư mục test
└───web                    # Thư mục chứa giao diện web
    ├───model              # Mô hình dự đoán
    ├───resource           # Tài nguyên web tĩnh
    └───__pycache__        # Thư mục cache Python
```

## Góp ý
Mọí góp ý và pull request luôn được hoan nghênh. Hãy đảm bảo rằng bạn đã chạy và kiểm tra mã nguồn trước khi gửi pull request!
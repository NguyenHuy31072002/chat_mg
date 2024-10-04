# 💻 Ứng dụng Chatbot Local Machine GenAI với Streamlit, MongoDB Atlas, Langchain và Gemini

'Nhấp chuột phải' để xem Video demo ⬇️ trong tab trình duyệt mới:

<!-- markdownlint-disable-next-line MD033 -->
<a href="https://youtu.be/E0RpmGbmKEg" target="_blank">
<!-- markdownlint-disable-next-line MD033 -->
  <img src="https://img.youtube.com/vi/E0RpmGbmKEg/0.jpg" alt="Watch the video">
</a>

---
<!-- markdownlint-disable-next-line MD033 -->
<span style="color:darkblue">_version = 1.0.0_</span>
<!-- markdownlint-enable MD033 -->

Ứng dụng Python Retrieval-Augmented Generation (RAG) này có thể đọc nhiều tệp PDF - tối đa 200MB cùng một lúc - và trả lời các câu hỏi dựa trên thông tin trong các tệp PDF đó. Nói một cách đơn giản hơn, nó có thể tìm thông tin có liên quan từ các tệp PDF và sử dụng thông tin đó để trả lời các câu hỏi của bạn.

Nó được phát triển cục bộ để triển khai trên nền tảng đám mây trong tương lai ☁️ - trên AWS và GCP - bằng cách sử dụng Serverless Containers. Ngăn xếp ứng dụng:
* **Streamlit** - Front End
* **Gemini** - LLM/Foundation Model
* **Langchain** - NLP Orchestration
* **MongoDB Atlas Vector Search** - Cloud-based Vector Database
* **Dotenv** - Local secret management
* **PyPDF** - PDF text extraction
* **PyTesseract** - OCR on AES Encrypted PDFs or PDFs with images in the background that would result in an empty text extraction
</br>

## Các tính năng chính

* Kết nối khóa API/TOKEN an toàn được ẩn trong .envtệp
* Xử lý nhiều tệp - lên đến 200MB - trong 1 thao tác tải lên duy nhất
* Khả năng trả lời các câu hỏi dựa trên các tài liệu đã được vector hóa và lưu trữ trong cơ sở dữ liệu - không cần phải tải lại các tệp PDF tương tự
* Khả năng trích xuất văn bản từ các tệp PDF bị khóa AES hoặc các tệp PDF có hình ảnh nền chặn việc trích xuất văn bản đơn giản
* Xử lý song song trích xuất văn bản cho PDF > 5MB để có hiệu suất ứng dụng nhanh hơn
* Nút 'Xóa lịch sử trò chuyện'
* Một loạt các tính năng quan sát/nhật ký để cân nhắc phát triển đám mây trong tương lai:
    * Hàm Langchain callbacktính toán mức sử dụng mã thông báo 'OpenAi' và in ra tệp nhật ký.
  ![cost-screenshot](images/openai-token-usage-mdb-logs-screenshot.png) 
  * Nhật ký hoạt động cụ thể của MongoDB được ghi lại thông qua pymongotrình điều khiển
  * Một `script execution time`chức năng đo lường

</br>

![mdb-vector-screenshot-1](images/mdb-compass-screenshot-1.png)

---

## Kiến trúc tham khảo

![architecture-diagram](images/local-rag-mdb-diagram.png)

---

## Điều kiện tiên quyết

* Python >=3.11
* Tesseract CLI
* OpenAI API Key
* MongoDB Atlas Cluster and Database

---

## Hướng dẫn thiết lập

Dưới đây là các bước hướng dẫn cách thiết lập ứng dụng này trên máy cục bộ của bạn ⬇️

#### MongoDB Atlas Setup

* Hướng dẫn thiết lập tài khoản, cụm và cơ sở dữ liệu MongoDB Atlas miễn phí có thể được tìm thấy [here](https://www.mongodb.com/docs/atlas/getting-started/).
* Sau khi cụm và cơ sở dữ liệu của bạn được thiết lập, hãy điều hướng đến Dịch vụ dữ liệu > Truy cập mạng và nhấp vào IP `Access List`. Xác nhận rằng địa chỉ IP của máy cục bộ của bạn đang ở `Active`Trạng thái.
 </br>
  
![networking-screenshot](images/mdb-networking-screenshot.png)
</br>

* Vào Dịch vụ dữ liệu > Cơ sở dữ liệu, nhấp vào tab `connect` và sao chép chuỗi kết nối cơ sở dữ liệu vào tệp `.env` của bạn.

</br>

![connection-string](images/mdb-connection-screenshot.png)
</br>

#### Tạo chỉ mục tìm kiếm vector

Bước cuối cùng sau khi tạo cụm, cơ sở dữ liệu và thiết lập kết nối là tạo tệp `vector search index`.

* Điều hướng đến Dịch vụ dữ liệu > Cụm của bạn > Duyệt bộ sưu tập > Tìm kiếm Atlas

* Chọn cơ sở dữ liệu bạn đã tạo cho dự án này - ví dụ, `chatbot_db`  sao chép và dán đoạn mã JSON bên dưới.

![json-vector-index](images/mdb-index-json-screenshot.png)

```json

{
  "fields": [
    {
      "numDimensions": 748,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}

```

* Chọn 'Tiếp theo'
* Xác nhận việc tạo chỉ mục bằng cách nhấp vào **Create Search Index**
* Chỉ mục sẽ sẵn sàng để sử dụng khi ở trạng thái 'Active Status'_

#### Thiết lập bổ sung

* Cài đặt [tesseract cli](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html)trên máy cục bộ của bạn và thêm `tesseract location path` vào file `.env` - `pytesseract` là một gói python cho `tesseract`, tuy nhiên, nó hoạt động ngoài tesseract cli được cài đặt cục bộ
* Chèn biến môi trường của bạn vào file `.env` . Để tham khảo, hãy xem [sample-dotenv-file.txt](sample-dotenv-file.txt) trong kho lưu trữ này.

#### Kích hoạt môi trường ảo

* Tạo `chatbot-app` môi trường ảo cho dự án của bạn: 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`python -m venv chatbot-env`
* Kích hoạt nó: 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`source chatbot-env/bin/activate`
* Cài đặt các phần phụ thuộc:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`pip install -r requirements.txt`  

#### Tập lệnh Shell

Ngoài ra, bạn có thể chỉnh sửa tập `sample_run_chatbot.sh` lệnh bash bằng thư mục dự án máy cục bộ của mình và chạy tập lệnh này để kích hoạt venv và chạy streamlit hoặc trong trường hợp venv được kích hoạt, để chạy streamlit:

`./sample_run_chatbot.sh`

---

## Cải tiến trong tương lai

* Tạo chức năng 'Nhập URL web' để người dùng có tùy chọn tải tệp lên hoặc thêm URL web PDF
* Triển khai trích xuất siêu dữ liệu PDF. Tạo tệp JSON siêu dữ liệu 'tài liệu đã tải lên' sẽ được gửi vào Cơ sở dữ liệu Atlas MongoDB riêng biệt để có bản ghi về tất cả các tệp PDF đã được vector hóa trước đó. Theo cách đó, người dùng sẽ có thể xem danh sách các tệp PDF này và đặt câu hỏi về chúng
* Tạo hộp thả xuống trong UI để người dùng có thể xem các tên tệp PDF có sẵn này
* Triển khai Cloud Native trên AWS và GCP

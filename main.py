from model import get_yolov5
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from fastapi.responses import Response
import cv2
import numpy as np

app = FastAPI(title="PIPE COUNTING API",
    description="""Upload PIPE image and the API will response number of pipe""",
    version="0.0.1",)

## สร้าง object ของโมเดลไว้สำหรับการเช็คโลโก้ในภาพ
model_pipe = get_yolov5(0.5)

@app.post("/detectImage")
async def detect_image(file: UploadFile):
    # Load your image using OpenCV
    image_data = await file.read()
    image_np = np.frombuffer(image_data, np.uint8)
    original_img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    
    # Inference with your model
    results = model_pipe(original_img, size=640)
    results.render(labels=True)
    data = results.pandas().xyxy[0]

    # Calculate the total count and display it in the top bar
    total_count = len(data)
    font_scale = 1.5
    font_thickness = 1
    font_color = (0, 0, 0)  # Black color
    background_color = (0, 0, 0)  # Black background color
    totals_text = f"Total: {total_count}"

    # Loop through the results and plot text with row numbers and a stroke inside each object
    for idx, row in data.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        text = str(idx + 1)  # Adding 1 to start row numbers from 1

        # Calculate the font size based on the size of the bounding box
        font_scale = min((xmax - xmin), (ymax - ymin)) / 100  # Adjust the divisor as needed

        # Calculate the position to center the text
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2

        # Draw the stroke (border) around the text
        for x_offset in [-1, 1]:
            for y_offset in [-1, 1]:
                cv2.putText(
                    original_img,
                    text,
                    (text_x + x_offset, text_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255,255,255),
                    font_thickness + 1,  # Adjust the thickness of the stroke
                    cv2.LINE_AA,
                )

        # Draw the text on the image
        cv2.putText(original_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
    # Create a top bar and add the text to it
    top_bar_height = 50  # Height of the top bar for displaying totals
    top_bar = np.zeros((top_bar_height, original_img.shape[1], 3), dtype=np.uint8)
    top_bar = cv2.rectangle(top_bar, (0, 0), (original_img.shape[1], top_bar_height), background_color, -1)
    cv2.putText(top_bar, totals_text, (20, top_bar_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), font_thickness)
    
    # Combine the top bar with the original image
    combined_img = np.vstack((top_bar, original_img))

    # Convert the combined image back to bytes
    _, img_bytes = cv2.imencode('.jpg', combined_img)
    img_bytes_io = BytesIO(img_bytes.tobytes())

    # Return the modified image as a response
    return Response(content=img_bytes_io.getvalue(), media_type="image/jpeg")
        

@app.post("/getLabel")
async def detect_image_label(file: UploadFile):
  
    img = Image.open(BytesIO(await file.read()))
    results = model_pipe(img, size= 1920)
   
    ## หลังจากที่ได้ผลลัพธ์จากโมเดลแล้ว เราจะใช้คำสั่ง .pandas().xyxy เพื่อได้ผลลัพธ์ออกมาในรูปแบบของ dataframe ว่ามีโลโก้อะไรบ้าง
    ## แล้วเราจะเลือกโลโก้และค่า confident ที่สูงที่สุด (กรณีมีหลายอัน) และแปลงออกมาเป็น list ส่งกลับไปให้ผู้ใช้งาน
    print(results.pandas().xyxy[0])
    label_result =  len(results.pandas().xyxy[0])#results.pandas().xyxy[0].groupby('name')[['confidence']].max().reset_index().values.tolist() 
    return {"label": label_result}
import cv2
import os
import logging
from telegram import Update, InputMediaPhoto
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Set up logging to track issues
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Load pre-trained DNN face detection model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Function to detect faces using DNN
def extract_faces_dnn(video_path, max_faces=10):
    cap = cv2.VideoCapture(video_path)
    faces_found = []
    frame_count = 0

    while cap.isOpened() and len(faces_found) < max_faces:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Process every 17th frame to save computation time
        if frame_count % 17 != 0:
            continue

        # Prepare the frame for DNN input
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123), swapRB=False, crop=False)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections based on confidence threshold (e.g., 0.5)
            if confidence > 0.5:
                # Get bounding box for face detection
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Crop and store the detected face
                face_frame = frame[startY:endY, startX:endX]
                faces_found.append(face_frame)
                if len(faces_found) >= max_faces:
                    break

    cap.release()
    return faces_found

# Handler for video messages
async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.video:
        video = update.message.video
        file = await context.bot.get_file(video.file_id)
        video_path = "input_video.mp4"

        # Ensure no directory conflicts
        if os.path.exists(video_path):
            os.remove(video_path)

        await file.download_to_drive(video_path)  # Save the video file
        await update.message.reply_text("Processing video to extract faces...")

        # Ask user how many faces to extract
        await update.message.reply_text("How many faces do you want to extract? (Max 10)")

        # Capture the user's response for number of faces
        def capture_faces(update: Update):
            try:
                max_faces = int(update.message.text)
                if max_faces > 10:
                    max_faces = 10
                    update.message.reply_text("Maximum limit is 10 faces.")
            except ValueError:
                max_faces = 5
                update.message.reply_text("Invalid input. Defaulting to 5 faces.")
            return max_faces

        max_faces = capture_faces(update)

        # Extract faces from the video
        if os.path.exists(video_path):
            faces = extract_faces_dnn(video_path, max_faces=max_faces)  # Use DNN-based extraction
            os.remove(video_path)  # Clean up the input file

            if faces:
                # Save and send up to 10 detected faces
                media_group = []
                for i, face in enumerate(faces):
                    face_path = f"detected_face_{i}.jpg"
                    cv2.imwrite(face_path, face)
                    media_group.append(InputMediaPhoto(media=open(face_path, 'rb')))
                
                # Send all detected faces as a media group
                await update.message.reply_media_group(media_group)

                # Clean up saved images
                for i in range(len(faces)):
                    os.remove(f"detected_face_{i}.jpg")
            else:
                await update.message.reply_text("No faces detected in the video. Please try another video.")
        else:
            await update.message.reply_text("Failed to download the video. Please try again.")
    else:
        await update.message.reply_text("Please send a valid video.")

# Start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! Send me a video, and I'll extract faces from it.")

# Set Faces command handler (Optional for future extension)
async def set_faces(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("You can set the number of faces, and it will be processed accordingly.")

# Main function to run the bot
def main():
    # Bot token replaced with the real one
    bot_token = "6934514903:AAHLVkYqPEwyIZiyqEhJocOrjDYwTk9ue8Y"  # Replace with your actual bot token

    app = Application.builder().token(bot_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("set_faces", set_faces))  # Optional command for future
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))

    # Thread pool executor for better performance with large files
    with ThreadPoolExecutor() as executor:
        app.run_polling(allowed_updates=Update.ALL_TYPES, poll_interval=1)

if __name__ == "__main__":
    main()

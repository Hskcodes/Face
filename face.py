import cv2
import os
from telegram import Update, InputMediaPhoto
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Function to detect faces and return multiple cropped faces with large padding
def extract_faces(video_path, max_faces=5, padding=50):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    faces_found = []
    frame_count = 0

    while cap.isOpened() and len(faces_found) < max_faces:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Process every 10th frame to save computation time
        if frame_count % 17 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Add larger padding to the face region
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)

            face_frame = frame[y1:y2, x1:x2]
            faces_found.append(face_frame)
            if len(faces_found) >= max_faces:
                break

    cap.release()
    return faces_found

# Handler for video messages
async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    video = update.message.video

    if video:
        file = await context.bot.get_file(video.file_id)
        video_path = "input_video.mp4"

        # Ensure no directory conflicts
        if os.path.exists(video_path):
            os.remove(video_path)

        await file.download_to_drive(video_path)  # Save the video file
        await update.message.reply_text("Processing video to extract faces...")

        # Extract faces from the video
        if os.path.exists(video_path):
            faces = extract_faces(video_path, max_faces=5, padding=150)  # Increase padding for larger zoom-out
            os.remove(video_path)  # Clean up the input file

            if faces:
                # Save and send up to 5 detected faces
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

# Main function to run the bot
def main():
    # Replace 'YOUR_TELEGRAM_BOT_TOKEN' with your bot token from BotFather
    bot_token = "BOTOKEN"

    app = Application.builder().token(bot_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))

    app.run_polling()

if __name__ == "__main__":
    main()


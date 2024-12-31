import cv2
import os
import logging
from telegram import Update, InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from concurrent.futures import ThreadPoolExecutor

# Set up logging to track issues
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

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

        # Process every 17th frame to save computation time
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
    if update.message and update.message.video:
        video = update.message.video
        file = await context.bot.get_file(video.file_id)
        video_path = "input_video.mp4"

        # Ensure no directory conflicts
        if os.path.exists(video_path):
            os.remove(video_path)

        await file.download_to_drive(video_path)  # Save the video file
        await update.message.reply_text("Processing video to extract faces...")

        # Ask user how many faces to extract (limit 10)
        keyboard = [[InlineKeyboardButton(str(i), callback_data=f'faces_{i}') for i in range(1, 11)]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("How many faces would you like to extract? (1 to 10)", reply_markup=reply_markup)

# Callback handler for inline button presses (user selects number of faces)
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # Acknowledge the button press
    
    choice = query.data
    if choice.startswith('faces_'):
        num_faces = int(choice.split('_')[1])
        context.user_data['max_faces'] = num_faces
        await query.edit_message_text(f"Number of faces to extract: {num_faces}")
        
        # Now process the video to extract faces
        video_path = "input_video.mp4"
        faces = extract_faces(video_path, max_faces=num_faces, padding=150)  # Using padding as per previous setup
        os.remove(video_path)  # Clean up the input file

        if faces:
            # Save and send up to the selected number of detected faces
            media_group = []
            for i, face in enumerate(faces):
                face_path = f"detected_face_{i}.jpg"
                cv2.imwrite(face_path, face)
                media_group.append(InputMediaPhoto(media=open(face_path, 'rb')))
            
            # Send all detected faces as a media group
            await query.message.reply_media_group(media_group)

            # Clean up saved images
            for i in range(len(faces)):
                os.remove(f"detected_face_{i}.jpg")
        else:
            await query.message.reply_text("No faces detected in the video. Please try another video.")

# Start command handler with emoji
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hi! Send me a video, and I'll extract faces from it.")

# Main function to run the bot
def main():
    # Hardcoded bot token (for testing purposes only)
    bot_token = "YOUR_BOT_TOKEN_HERE"  # Replace with your actual bot token

    app = Application.builder().token(bot_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))
    app.add_handler(CallbackQueryHandler(button))  # Add the callback query handler for button presses

    # Thread pool executor for better performance with large files
    with ThreadPoolExecutor() as executor:
        app.run_polling(allowed_updates=Update.ALL_TYPES, poll_interval=1)

if __name__ == "__main__":
    main()

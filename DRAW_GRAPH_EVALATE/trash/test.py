import cv2

# Open a video file
cap = cv2.VideoCapture(1)

# Check if the video file was opened successfully
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("End of video.")
        break

    # Encode the frame as PNG
    _, encoded_image = cv2.imencode('.jpg', frame)

    # Decode the encoded PNG image
    decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

    # Display the decoded image
    cv2.imshow('Decoded Image', decoded_image)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

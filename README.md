# some-face-recognizer
Face recognition system. It perform face detection, extract face embeddings from each face using deep learning, train a face recognition model on the embeddings, and then finally recognize faces.

__Using__
1) Run gather_face_dataset/build_face_dataset.py script to save some face screenshots, for model to learn.
2) Use extract_embeddings.py to get from images embeddings for network model.
3) Run train_model.py to train existing network model.
4) And, finaly, enjoy face recognition with image_recogize.py, or video_recognize.py or, even video from WebCam using detect_faces_video.py.

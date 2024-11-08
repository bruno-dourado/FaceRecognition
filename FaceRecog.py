import glob
import os
import cv2
import numpy as np
import face_recognition

class FaceRecog:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Redimensionando frame para melhorar velocidade 
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        # Carregar Imagens com os Rostos da pasta indicada
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} imagens encontradas.".format(len(images_path)))

        # Armazenar a codificação das imagens e os nomes
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Armazenando o nome do arquivo e a codificação
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Imagens carregadas")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Encontrar todas as faces no frame do vídeo
        # Converter a imagem de BGR para (utilizada pelo OpenCV) para RGB (utilizada pela face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Verifica se a face encontrada casa com as faces conhecidas
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Desconhecido"

            # Utilizar a face conhecida com a menor distancia ao invés de trabalhar apenas com a primeira face encontrada
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convertendo para um array para ajuste do rapido do quadro
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

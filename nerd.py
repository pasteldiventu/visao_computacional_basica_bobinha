import cv2
import mediapipe as mp
import os
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


IMAGE_FOLDER = 'imagens'
try:
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(valid_extensions)]
except FileNotFoundError:
    print(f"Erro: A pasta '{IMAGE_FOLDER}' não foi encontrada.")
    print("Certifique-se de que ela existe no mesmo local que o script.")
    exit()
try:
    img_to_show = cv2.imread('imagem_gesto.png')
    img_to_show = cv2.resize(img_to_show, (400, 400))
except:
    print("Erro: Não foi possível carregar a 'imagem_gesto.png'.")
    print("Certifique-se de que a imagem está na mesma pasta que o script.")
    exit()

WINDOW_NAME = 'Nerd'

cap = cv2.VideoCapture(0)

image_window_open = False

print("Iniciando a webcam... Pressione 'ESC' para sair.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignorando frame vazio da câmera.")
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1280, 720))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    gesture_detected = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

            try:
                index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                index_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y

                middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                middle_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

                ring_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
                ring_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
                
                pinky_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
                pinky_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

                is_index_up = index_tip_y < index_pip_y

                is_middle_down = middle_tip_y > middle_pip_y
                is_ring_down = ring_tip_y > ring_pip_y
                is_pinky_down = pinky_tip_y > pinky_pip_y

                if is_index_up and is_middle_down and is_ring_down and is_pinky_down:
                    gesture_detected = True

            except Exception as e:
                print(f"Erro ao processar landmarks: {e}")

    if gesture_detected:
        if not image_window_open:
            try:
                # 1. Sorteia um nome de arquivo da lista
                random_image_name = random.choice(image_files)
                # 2. Monta o caminho completo (pasta + nome do arquivo)
                image_path = os.path.join(IMAGE_FOLDER, random_image_name)
                # 3. Carrega a imagem sorteada
                img_to_show = cv2.imread(image_path)
                # 4. Redimensiona para um tamanho fixo (ex: 400x400)
                img_to_show = cv2.resize(img_to_show, (400, 400))
                
                # 5. Mostra a imagem
                cv2.imshow(WINDOW_NAME, img_to_show)
                image_window_open = True
                
            except Exception as e:
                print(f"Erro ao carregar ou mostrar imagem {image_path}: {e}")
    else:
        if image_window_open:
            cv2.destroyWindow(WINDOW_NAME)
            image_window_open = False

    cv2.imshow('Webcam - Detector de Gesto', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
hands.close()
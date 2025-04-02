import cv2
from ultralytics import YOLO
modelo = YOLO('yolov8n.pt')

cv2.namedWindow("Webcam")
cv2.namedWindow("Tela Plástico")
cv2.namedWindow("Tela Metal")

def atualizar_telas(rotulos_detectados):
  
    if 'bottle' in rotulos_detectados:
        tela_bottle = cv2.imread('test/aberto_plastico.png')  
    else:
        tela_bottle = cv2.imread('test/fechado_plastico.png')
  
    if 'fork' in rotulos_detectados:
        tela_fork = cv2.imread('test/aberto_metal.png')  
    else:
        tela_fork = cv2.imread('test/fechado_metal.png') 

      if tela_bottle is not None:
        cv2.imshow("Tela Plástico", tela_bottle)
    else:
        print("Erro ao carregar a imagem para Plástico")
    
    if tela_fork is not None:
        cv2.imshow("Tela Metal", tela_fork)
    else:
        print("Erro ao carregar a imagem para Metal.")

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o quadro da webcam.")
        break

    resultados = modelo.predict(source=frame, conf=0.5, verbose=False)
 
    rotulos_detectados = set()

    for resultado in resultados:
        for objeto in resultado.boxes.data.tolist():
            classe_id = int(objeto[-1])  
            rotulo = resultado.names[classe_id]
            rotulos_detectados.add(rotulo)

            x1, y1, x2, y2 = int(objeto[0]), int(objeto[1]), int(objeto[2]), int(objeto[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, rotulo, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    atualizar_telas(rotulos_detectados)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

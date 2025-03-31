Para implementar uma simulação com duas telas que exibem os estados "Aberto" ou "Fechado" dependendo do rótulo identificado ("person" ou "car"), você pode usar uma biblioteca de interface gráfica como o **OpenCV** para exibir essas telas. Aqui está uma adaptação do código anterior com a funcionalidade de simulação:

```python
import cv2
from ultralytics import YOLO

# Carrega o modelo YOLO
modelo = YOLO('yolov8n.pt')

# Inicializa as janelas
cv2.namedWindow("Tela para 'person'")
cv2.namedWindow("Tela para 'car'")

def atualizar_telas(rotulos_detectados):
    # Atualiza a tela para 'person'
    if 'person' in rotulos_detectados:
        tela_person = cv2.imread('aberto_person.png')  # imagem de "Aberto"
    else:
        tela_person = cv2.imread('fechado_person.png')  # imagem de "Fechado"
    
    # Atualiza a tela para 'car'
    if 'car' in rotulos_detectados:
        tela_car = cv2.imread('aberto_car.png')  # imagem de "Aberto"
    else:
        tela_car = cv2.imread('fechado_car.png')  # imagem de "Fechado"
    
    # Exibe as telas
    cv2.imshow("Tela para 'person'", tela_person)
    cv2.imshow("Tela para 'car'", tela_car)

# Loop para detecção em tempo real
while True:
    resultados = modelo.predict(source='0', show=True)
    rotulos_detectados = set()
    
    for resultado in resultados:
        for objeto in resultado.boxes.data.tolist():
            # Pega o rótulo do objeto detectado
            rotulo = resultado.names[int(objeto[-1])]
            rotulos_detectados.add(rotulo)
    
    # Atualiza as telas baseado nos rótulos detectados
    atualizar_telas(rotulos_detectados)

    # Adiciona tecla para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cv2.destroyAllWindows()
```

### Explicação:
1. **Imagens "Aberto" e "Fechado"**: Use imagens PNG personalizadas para simular os estados das telas. Certifique-se de salvar as imagens com os nomes `aberto_person.png`, `fechado_person.png`, `aberto_car.png` e `fechado_car.png` no mesmo diretório do script.

2. **Função `atualizar_telas(rotulos_detectados)`**: Essa função verifica quais rótulos foram detectados e atualiza as telas exibindo as imagens apropriadas.

3. **Controle de loop**: O código roda em tempo real e atualiza os estados das telas conforme os rótulos são detectados pela câmera.

Você pode adaptar os nomes das imagens ou os rótulos para atender às suas necessidades. Se precisar de ajuda com algum detalhe, estarei aqui para colaborar! 😊

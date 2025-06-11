# Stereo Z-Scan

## Descrição

O **Stereo Z-Scan** é um projeto para reconstrução 3D de alta resolução utilizando sistemas de câmeras estéreo. Ele oferece ferramentas para captura, processamento e análise de imagens estéreo, permitindo a geração de modelos 3D detalhados a partir de pares de imagens 2D. O processamento é acelerado por GPU (CUDA), tornando o fluxo eficiente mesmo para grandes volumes de dados.

---

## Principais Funcionalidades

- **Captura de Imagens Estéreo**: Integração com sistemas de câmeras estéreo para aquisição sincronizada de imagens.
- **Reconstrução 3D**: Algoritmos para estimação de profundidade e geração de nuvens de pontos 3D.
- **Visualização**: Ferramentas para visualização e análise dos resultados em 3D.
- **Aceleração CUDA**: Uso de GPU para acelerar etapas críticas do processamento.
- **Filtros Espaciais e Temporais**: Métodos para correlação espaço-temporal e filtragem de pontos esparsos.

---

## Requisitos

### Hardware
- Sistema de câmeras estéreo (ex: Spinnaker)
- GPU compatível com CUDA (recomendado)

### Software
- Python 3.x
- OpenCV 4.x
- [Spinnaker SDK for Python](https://www.teledynevisionsolutions.com/products/spinnaker-sdk/?model=Spinnaker%20SDK&vertical=machine%20vision&segment=iis)
- Cupy (para aceleração CUDA)
- Matplotlib, NumPy, SciPy

---

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/Projeto-Voris/stereo_zscan.git
   cd stereo_zscan
  ```
2. Instale dependências
  ```bash
  pip install -r requirements.txt
  ```

## Uso
1. Ajuste os arquivos de configuração em ```cfg/``` conforme seu sistema de câmeras e parâmetros de calibração.
2. Execute o script principal para processar as imagens e gerar a reconstrução 3D:
```bash
  python main_spatial.py
```
3. Os resultados (nuvens de pontos, mapas de profundidade, etc.) serão salvos na pasta de saída definida no script.

## Organização do Projeto

* main_spatial.py: Pipeline principal para reconstrução 3D.
* include/SpatialCorrelation.py: Algoritmos de correlação espacial e temporal.
* extras/: Scripts auxiliares para visualização, depuração e pré-processamento.
* cfg/: Arquivos de configuração de calibração e parâmetros do sistema.
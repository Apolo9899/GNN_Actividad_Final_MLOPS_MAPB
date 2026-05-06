# GNN Lipophilicity — Proyecto MLOps

**Autor:** Marco Apolo Pulpillo Berrocal  
**Máster Deep Learning — Universidad Politécnica de Madrid**

Predicción de lipofilicidad molecular mediante redes neuronales de grafos (GNN), con API REST, contenedorización Docker y CI/CD con GitHub Actions.

---

## Funcionalidades principales

- **Entrenamiento** de múltiples arquitecturas GNN (GCN, GIN, GAT, GIN-JK, VirtualNodeGIN) con logging en Weights & Biases
- **API REST** (FastAPI) para predicción de lipofilicidad a partir de un grafo molecular en JSON
- **Tests automatizados** con pytest (modelos y API)
- **Contenedor Docker** listo para despliegue
- **CI/CD** con GitHub Actions (tests + lint + build Docker)

---

## Configuración del entorno local

### Requisitos
- Python 3.10+
- Docker (para el contenedor)

### Instalación

```bash
git clone <URL_REPO>
cd gnn-lipophilicity-mlops

python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install -r requirements.txt
```

### Lanzar la API en local

```bash
uvicorn src.inference_api:app --reload --port 8000
```

Documentación interactiva: `http://localhost:8000/docs`

### Ejecutar los tests

```bash
pytest tests/ -v
```

### Entrenamiento con W&B

```bash
wandb login
python src/train.py --model gin_jk --epochs 600 --wandb --wandb_project gnn-lipophilicity
```

### Docker

```bash
docker build -t gnn-lipophilicity:latest .
docker run -p 8000:8000 gnn-lipophilicity:latest
```

---

## Enlaces

- **GitHub:** https://github.com/Apolo9899/GNN_Actividad_Final_MLOPS_MAPB
- **Weights & Biases:** https://wandb.ai/marcoapolo-upm/gnn-lipophilicity/reports/GNN-Lipophilicity-—-Análisis-de-Experimentos--VmlldzoxNjc4ODMxMg==
- **Endpoint en producción:** 
Online: 
	https://gnnimage-latest.onrender.com/docs

Local:
	docker run -p 8000:8000 gnnimage
	http://localhost:8000/docs #realizado mediante Docker

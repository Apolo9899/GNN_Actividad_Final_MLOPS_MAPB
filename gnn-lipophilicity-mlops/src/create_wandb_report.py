"""
Crea automáticamente el W&B Report del proyecto GNN Lipophilicity.

Uso:
    python src/create_wandb_report.py --project gnn-lipophilicity --entity TU_USUARIO
"""
import argparse

import wandb
import wandb_workspaces.reports.v2 as wr


def create_report(project: str, entity: str):
    report = wr.Report(
        project=project,
        entity=entity,
        title="GNN Lipophilicity — Análisis de Experimentos",
        description=(
            "Comparativa de arquitecturas GNN para predicción de lipofilicidad molecular. "
            "Proyecto MLOps — Máster Deep Learning UPM."
        ),
    )

    runset_all = wr.Runset(
        project=project,
        entity=entity,
        name="Todos los experimentos",
    )

    report.blocks = [
        # ── 1. INTRODUCCIÓN ──────────────────────────────────────────────────
        wr.H1(text="1. Introducción"),
        wr.P(
            text=(
                "Este experimento compara distintas arquitecturas de **Graph Neural Networks (GNN)** "
                "para predecir la **lipofilicidad** (log P) de moléculas representadas como grafos atómicos. "
                "La lipofilicidad mide la afinidad de una molécula por entornos lipídicos y es una propiedad "
                "clave en el diseño de fármacos.\n\n"
                "**Dataset:** 4 200 moléculas en formato GML (3 360 entrenamiento / 840 test). "
                "Cada nodo del grafo representa un átomo con 9 features. "
                "La métrica de evaluación principal es el **Mean Squared Error (MSE)**."
            )
        ),

        # ── 2. TABLA COMPARATIVA ─────────────────────────────────────────────
        wr.H1(text="2. Comparativa de arquitecturas"),
        wr.P(
            text=(
                "La tabla siguiente muestra todos los runs con sus hiperparámetros clave "
                "y el mejor MSE de validación obtenido."
            )
        ),
        wr.PanelGrid(
            runsets=[runset_all],
            panels=[
                wr.RunComparer(
                    diff_only="split",
                    layout={"x": 0, "y": 0, "w": 24, "h": 9},
                ),
            ],
        ),

        # ── 3. CURVAS DE ENTRENAMIENTO ────────────────────────────────────────
        wr.H1(text="3. Curvas de entrenamiento"),
        wr.P(
            text=(
                "Evolución del MSE de validación y la pérdida de entrenamiento a lo largo de las épocas "
                "para los tres primeros experimentos (GCN, GIN, GAT). "
                "Se observa cómo el GIN con BatchNorm converge más rápido y a un MSE menor que el GCN baseline."
            )
        ),
        wr.PanelGrid(
            runsets=[runset_all],
            panels=[
                wr.LinePlot(
                    title="Val MSE por época",
                    x="epoch",
                    y=["val_mse"],
                    smoothing_factor=0.0,
                    layout={"x": 0, "y": 0, "w": 12, "h": 6},
                ),
                wr.LinePlot(
                    title="Train Loss por época",
                    x="epoch",
                    y=["train_loss"],
                    smoothing_factor=0.0,
                    layout={"x": 12, "y": 0, "w": 12, "h": 6},
                ),
            ],
        ),

        # ── 4. COMPARATIVA FINAL ──────────────────────────────────────────────
        wr.H1(text="4. Comparativa final de MSE"),
        wr.P(
            text=(
                "El gráfico de barras compara el mejor MSE de validación de cada arquitectura. "
                "Menor MSE indica mejor predicción."
            )
        ),
        wr.PanelGrid(
            runsets=[runset_all],
            panels=[
                wr.BarPlot(
                    title="Best Val MSE por arquitectura",
                    metrics=["best_val_mse"],
                    orientation="v",
                    layout={"x": 0, "y": 0, "w": 24, "h": 8},
                ),
            ],
        ),

        # ── 5. CONCLUSIONES ───────────────────────────────────────────────────
        wr.H1(text="5. Conclusiones"),
        wr.P(
            text=(
                "**Modelo ganador: GIN Avanzado con Jumping Knowledge + Multi-scale Pooling** "
                "(MSE de validación: 0.3030).\n\n"
                "Conclusiones principales:\n\n"
                "- El **GCN baseline** (MSE 0.7505) sirve como punto de referencia. "
                "Su arquitectura simple limita la capacidad expresiva.\n\n"
                "- El **GIN con BatchNorm** (MSE 0.4162) mejora significativamente gracias a la "
                "agregación suma (teóricamente más discriminativa según el test de Weisfeiler-Lehman) "
                "y la normalización por lotes.\n\n"
                "- El **GAT** (MSE 0.6392) no supera al GIN a pesar de usar mecanismos de atención, "
                "posiblemente porque la atención introduce ruido en grafos moleculares pequeños.\n\n"
                "- El **GIN Avanzado con JK + Multi-scale Pooling** (MSE 0.3030) es el mejor modelo gracias "
                "a tres técnicas combinadas: conexiones residuales que evitan el vanishing gradient, "
                "Jumping Knowledge que captura información a múltiples escalas de vecindad, y pooling "
                "multi-escala (add + mean + max) que enriquece la representación global del grafo.\n\n"
                "- El **K-Fold Ensemble** (MSE 0.3055 ± 0.011) confirma que el GIN Avanzado es "
                "robusto entre distintas particiones del dataset.\n\n"
                "- El **GIN con Nodo Virtual** y el **GraphMAE Fine-Tune** quedan por encima del "
                "GIN Avanzado, sugiriendo que la comunicación global explícita y el preentrenamiento "
                "auto-supervisado no aportan mejoras en este dataset de tamaño moderado."
            )
        ),
    ]

    report.save()
    print(f"\n✅ Report creado correctamente.")
    print(f"🔗 URL: {report.url}\n")
    return report.url


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="gnn-lipophilicity")
    parser.add_argument(
        "--entity",
        type=str,
        required=True,
        help="Tu nombre de usuario en W&B (lo ves en wandb.ai/settings)",
    )
    args = parser.parse_args()

    create_report(args.project, args.entity)

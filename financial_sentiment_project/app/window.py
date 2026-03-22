from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.config import FIGURES_DIR, OUTPUT_DIR
from src.classical_models import run_classical_phase
from src.data_pipeline import run_phase1
from src.evaluate import run_evaluation_phase
from src.neural_model import run_neural_phase
from src.nosible_experiment import run_nosible_experiment


PHASE_LABELS = {
    "phase1": "Phase 1: Data Pipeline",
    "classical": "Phase 2: Classical Models",
    "neural": "Phase 3: GRU Model",
    "evaluate": "Phase 4: Evaluation",
    "all": "Run Full Pipeline",
    "nosible": "Extension: NOSIBLE Experiment",
    "raw_finbert": "Extension: Raw FinBERT",
    "finbert_finetuned": "Extension: Fine-tuned FinBERT",
}


class FigureScrollArea(QScrollArea):
    wheel_navigated = Signal(int)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        direction = -1 if delta > 0 else 1
        self.wheel_navigated.emit(direction)
        event.accept()


class PhaseWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, phase: str) -> None:
        super().__init__()
        self.phase = phase

    @Slot()
    def run(self) -> None:
        phase_map = {
            "phase1": run_phase1,
            "classical": run_classical_phase,
            "neural": run_neural_phase,
            "evaluate": run_evaluation_phase,
            "nosible": run_nosible_experiment,
        }
        try:
            if self.phase == "all":
                results: dict[str, object] = {}
                for phase_name in ("phase1", "classical", "neural", "evaluate"):
                    self.status.emit(f"Running {PHASE_LABELS[phase_name]}...")
                    results[phase_name] = phase_map[phase_name]()
                self.finished.emit(results)
                return

            if self.phase == "raw_finbert":
                from src.transformer_models import run_raw_finbert_phase

                self.status.emit(f"Running {PHASE_LABELS[self.phase]}...")
                self.finished.emit({self.phase: run_raw_finbert_phase()})
                return

            if self.phase == "finbert_finetuned":
                from src.transformer_models import run_finbert_finetuned_phase

                self.status.emit(f"Running {PHASE_LABELS[self.phase]}...")
                self.finished.emit({self.phase: run_finbert_finetuned_phase()})
                return

            self.status.emit(f"Running {PHASE_LABELS[self.phase]}...")
            result = phase_map[self.phase]()
            self.finished.emit({self.phase: result})
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.thread: QThread | None = None
        self.worker: PhaseWorker | None = None
        self.current_figure_path: Path | None = None
        self.current_figure_pixmap: QPixmap | None = None
        self.figure_paths: list[Path] = []

        self.setWindowTitle("Financial Sentiment Project")
        self.resize(1380, 860)
        self._build_ui()
        self.refresh_views()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        controls_panel = self._build_controls_panel()
        content_panel = self._build_content_panel()

        splitter.addWidget(controls_panel)
        splitter.addWidget(content_panel)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 980])

        layout = QHBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.addWidget(splitter)

        self.setStyleSheet(
            """
            QWidget {
                background: #f5f3ec;
                color: #1f1f1f;
                font-size: 13px;
            }
            QMainWindow {
                background: #efe9dd;
            }
            QGroupBox {
                border: 1px solid #d8d1c5;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 12px;
                font-weight: 600;
                background: #fbfaf6;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background: #214d41;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 9px 14px;
            }
            QPushButton:hover {
                background: #286051;
            }
            QPushButton:disabled {
                background: #93a59e;
            }
            QComboBox, QPlainTextEdit, QTableWidget {
                background: #fffdf8;
                border: 1px solid #d4ccbe;
                border-radius: 8px;
            }
            QLabel#Headline {
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#Subhead {
                color: #5d564b;
            }
            QLabel#CardValue {
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#CardLabel {
                color: #6a655c;
            }
            QTabWidget::pane {
                border: 1px solid #d4ccbe;
                border-radius: 10px;
                background: #fbfaf6;
            }
            QTabBar::tab {
                background: #e4ddcf;
                border: 1px solid #d4ccbe;
                padding: 8px 14px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background: #fbfaf6;
            }
            """
        )

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        headline = QLabel("Financial Sentiment Lab")
        headline.setObjectName("Headline")
        subtitle = QLabel("Run the core pipeline from `src/` and inspect metrics, summaries, and confusion matrices.")
        subtitle.setObjectName("Subhead")
        subtitle.setWordWrap(True)

        layout.addWidget(headline)
        layout.addWidget(subtitle)

        run_group = QGroupBox("Run Pipeline")
        run_layout = QVBoxLayout(run_group)

        self.phase_combo = QComboBox()
        for key, label in PHASE_LABELS.items():
            self.phase_combo.addItem(label, userData=key)

        self.run_button = QPushButton("Run Selected Phase")
        self.run_button.clicked.connect(self.run_selected_phase)

        self.refresh_button = QPushButton("Refresh Views")
        self.refresh_button.clicked.connect(self.refresh_views)

        self.open_outputs_button = QPushButton("Open Outputs Folder")
        self.open_outputs_button.clicked.connect(self.open_outputs_folder)

        run_layout.addWidget(self.phase_combo)
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.refresh_button)
        run_layout.addWidget(self.open_outputs_button)

        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(260)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.log_text)

        layout.addWidget(run_group)
        layout.addWidget(status_group)
        layout.addStretch(1)
        return panel

    def _build_content_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        cards_layout = QHBoxLayout()
        self.metrics_card = self._build_stat_card("Metrics Rows", "0")
        self.figures_card = self._build_stat_card("Figures", "0")
        self.models_card = self._build_stat_card("Result Files", "0")
        cards_layout.addWidget(self.metrics_card)
        cards_layout.addWidget(self.figures_card)
        cards_layout.addWidget(self.models_card)
        layout.addLayout(cards_layout)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_metrics_tab(), "Metrics")
        self.tabs.addTab(self._build_figures_tab(), "Figures")
        self.tabs.addTab(self._build_json_tab(), "Summaries")
        layout.addWidget(self.tabs)

        return panel

    def _build_stat_card(self, label_text: str, value_text: str) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            "QFrame { background: #fbfaf6; border: 1px solid #d8d1c5; border-radius: 12px; }"
        )
        layout = QVBoxLayout(frame)
        value = QLabel(value_text)
        value.setObjectName("CardValue")
        name = QLabel(label_text)
        name.setObjectName("CardLabel")
        layout.addWidget(value)
        layout.addWidget(name)
        frame.value_label = value  # type: ignore[attr-defined]
        return frame

    def _build_metrics_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.metrics_table = QTableWidget()
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.verticalHeader().setVisible(False)
        layout.addWidget(self.metrics_table)
        return tab

    def _build_figures_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top_row = QHBoxLayout()
        self.figure_combo = QComboBox()
        self.figure_combo.currentIndexChanged.connect(self.show_selected_figure)
        self.open_figure_button = QPushButton("Open Figure...")
        self.open_figure_button.clicked.connect(self.open_external_figure)
        top_row.addWidget(self.figure_combo, 1)
        top_row.addWidget(self.open_figure_button)

        self.figure_scroll = FigureScrollArea()
        self.figure_scroll.setWidgetResizable(True)
        self.figure_scroll.wheel_navigated.connect(self.cycle_figure)
        self.figure_label = QLabel("No figure selected.")
        self.figure_label.setAlignment(Qt.AlignCenter)
        self.figure_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.figure_label.setMinimumHeight(520)
        self.figure_scroll.setWidget(self.figure_label)

        layout.addLayout(top_row)
        layout.addWidget(self.figure_scroll)
        return tab

    def _build_json_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top_row = QHBoxLayout()
        self.json_combo = QComboBox()
        self.json_combo.currentIndexChanged.connect(self.show_selected_json)
        self.open_json_button = QPushButton("Open File...")
        self.open_json_button.clicked.connect(self.open_external_json)
        top_row.addWidget(self.json_combo, 1)
        top_row.addWidget(self.open_json_button)

        self.json_text = QPlainTextEdit()
        self.json_text.setReadOnly(True)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(10)
        self.json_text.setFont(mono)

        layout.addLayout(top_row)
        layout.addWidget(self.json_text)
        return tab

    @Slot()
    def run_selected_phase(self) -> None:
        if self.thread is not None:
            return

        phase = self.phase_combo.currentData()
        self.run_button.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.status_label.setText(f"Starting {PHASE_LABELS[phase]}...")
        self.log_text.appendPlainText(f"> {PHASE_LABELS[phase]}")

        self.thread = QThread(self)
        self.worker = PhaseWorker(phase)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self.on_worker_status)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_worker)

        self.thread.start()

    @Slot(str)
    def on_worker_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.log_text.appendPlainText(message)

    @Slot(dict)
    def on_worker_finished(self, result: dict) -> None:
        self.status_label.setText("Completed successfully.")
        self.log_text.appendPlainText(json.dumps(result, indent=2))
        self.refresh_views()

    @Slot(str)
    def on_worker_failed(self, error_text: str) -> None:
        self.status_label.setText("Run failed.")
        self.log_text.appendPlainText(error_text)
        QMessageBox.critical(self, "Pipeline Failed", error_text)

    @Slot()
    def cleanup_worker(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        if self.thread is not None:
            self.thread.deleteLater()
            self.thread = None
        self.run_button.setEnabled(True)
        self.refresh_button.setEnabled(True)

    @Slot()
    def refresh_views(self) -> None:
        self.load_metrics_table()
        self.load_figure_list()
        self.load_json_list()
        self.update_cards()
        self.status_label.setText("Views refreshed.")

    def load_metrics_table(self) -> None:
        metrics_dir = OUTPUT_DIR / "metrics"
        csv_specs = [
            ("main", metrics_dir / "evaluation_summary.csv"),
            ("nosible", metrics_dir / "nosible_evaluation_summary.csv"),
        ]

        frames: list[pd.DataFrame] = []
        for experiment_name, path in csv_specs:
            if path.exists():
                frame = pd.read_csv(path)
                frame.insert(0, "experiment", experiment_name)
                frames.append(frame)

        if not frames:
            self.metrics_table.clear()
            self.metrics_table.setRowCount(0)
            self.metrics_table.setColumnCount(0)
            return

        frame = pd.concat(frames, ignore_index=True)
        frame = frame.sort_values(["experiment", "split", "macro_f1"], ascending=[True, True, False]).reset_index(drop=True)
        self.metrics_table.setRowCount(len(frame))
        self.metrics_table.setColumnCount(len(frame.columns))
        self.metrics_table.setHorizontalHeaderLabels(frame.columns.tolist())

        for row_index, (_, row) in enumerate(frame.iterrows()):
            for col_index, column in enumerate(frame.columns):
                value = row[column]
                if isinstance(value, float):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                if column in {"accuracy", "macro_f1"}:
                    item.setTextAlignment(Qt.AlignCenter)
                self.metrics_table.setItem(row_index, col_index, item)

        self.metrics_table.resizeColumnsToContents()

    def load_figure_list(self) -> None:
        figure_paths = sorted(FIGURES_DIR.glob("*.png"))
        self.figure_paths = figure_paths
        current = self.figure_combo.currentData()
        self.figure_combo.blockSignals(True)
        self.figure_combo.clear()
        for path in figure_paths:
            self.figure_combo.addItem(path.name, userData=path)
        self.figure_combo.blockSignals(False)

        if figure_paths:
            index = 0
            if current is not None:
                for idx in range(self.figure_combo.count()):
                    if self.figure_combo.itemData(idx) == current:
                        index = idx
                        break
            self.figure_combo.setCurrentIndex(index)
            self.show_selected_figure()
        else:
            self.current_figure_path = None
            self.current_figure_pixmap = None
            self.figure_label.setText("No confusion matrix figures found yet.")
            self.figure_label.setPixmap(QPixmap())

    @Slot(int)
    def cycle_figure(self, direction: int) -> None:
        if not self.figure_paths or self.figure_combo.count() == 0:
            return
        current_index = self.figure_combo.currentIndex()
        if current_index < 0:
            current_index = 0
        next_index = (current_index + direction) % self.figure_combo.count()
        self.figure_combo.setCurrentIndex(next_index)

    def load_json_list(self) -> None:
        json_paths = sorted((OUTPUT_DIR / "summaries").glob("*.json")) + sorted((OUTPUT_DIR / "metrics").glob("*.json"))
        current = self.json_combo.currentData()
        self.json_combo.blockSignals(True)
        self.json_combo.clear()
        for path in json_paths:
            self.json_combo.addItem(str(path.relative_to(Path.cwd())), userData=path)
        self.json_combo.blockSignals(False)

        if json_paths:
            index = 0
            if current is not None:
                for idx in range(self.json_combo.count()):
                    if self.json_combo.itemData(idx) == current:
                        index = idx
                        break
            self.json_combo.setCurrentIndex(index)
            self.show_selected_json()
        else:
            self.json_text.setPlainText("No summary JSON files found yet.")

    def update_cards(self) -> None:
        metrics_rows = self.metrics_table.rowCount()
        figures_count = len(list(FIGURES_DIR.glob("*.png")))
        result_files = len(list((OUTPUT_DIR / "metrics").glob("*"))) + len(list((OUTPUT_DIR / "models").glob("*")))
        self.metrics_card.value_label.setText(str(metrics_rows))  # type: ignore[attr-defined]
        self.figures_card.value_label.setText(str(figures_count))  # type: ignore[attr-defined]
        self.models_card.value_label.setText(str(result_files))  # type: ignore[attr-defined]

    @Slot()
    def show_selected_json(self) -> None:
        path = self.json_combo.currentData()
        if path is None:
            self.json_text.setPlainText("No JSON file selected.")
            return
        try:
            content = path.read_text(encoding="utf-8")
            try:
                parsed = json.loads(content)
                content = json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                pass
            self.json_text.setPlainText(content)
        except Exception as exc:
            self.json_text.setPlainText(f"Failed to load {path}:\n{exc}")

    @Slot()
    def show_selected_figure(self) -> None:
        path = self.figure_combo.currentData()
        if path is None:
            self.figure_label.setText("No figure selected.")
            self.figure_label.setPixmap(QPixmap())
            return
        self.current_figure_path = path
        self.current_figure_pixmap = QPixmap(str(path))
        self._render_current_figure()

    def _render_current_figure(self) -> None:
        if self.current_figure_pixmap is None or self.current_figure_pixmap.isNull():
            self.figure_label.setText("Failed to load image.")
            self.figure_label.setPixmap(QPixmap())
            return
        viewport_width = max(self.figure_scroll.viewport().width() - 30, 200)
        scaled = self.current_figure_pixmap.scaledToWidth(viewport_width, Qt.SmoothTransformation)
        self.figure_label.setPixmap(scaled)
        self.figure_label.setText("")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.current_figure_pixmap is not None:
            self._render_current_figure()

    @Slot()
    def open_outputs_folder(self) -> None:
        os.startfile(str(OUTPUT_DIR))

    @Slot()
    def open_external_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open JSON File",
            str(OUTPUT_DIR),
            "JSON Files (*.json);;All Files (*.*)",
        )
        if path:
            os.startfile(path)

    @Slot()
    def open_external_figure(self) -> None:
        path = self.figure_combo.currentData()
        if path is not None:
            os.startfile(str(path))


def launch_app() -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()

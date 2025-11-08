# 🧬 AI Ring Designer

**Self-learning system voor het genereren van organische 3D-geprinte koppelingen**

Een volledig autonoom AI-systeem dat leert om esthetische en functionele koppelingsstukken te ontwerpen voor kartonnen buizen. Het systeem combineert Deep Learning (Neural Networks), Evolutionary Algorithms, en Reinforcement Learning om designs te verbeteren over tijd.

---

## 📋 Inhoudsopgave

- [Features](#-features)
- [Vereisten](#-vereisten)
- [Installatie](#-installatie)
- [Quick Start](#-quick-start)
- [Gebruik](#-gebruik)
- [Project Structuur](#-project-structuur)
- [Hoe het werkt](#-hoe-het-werkt)
- [Troubleshooting](#-troubleshooting)
- [Configuratie](#-configuratie)
- [Resultaten bekijken](#-resultaten-bekijken)

---

## ✨ Features

- 🤖 **Self-learning**: Het systeem verbetert zichzelf zonder menselijke input
- 📊 **Real-time visualisatie**: Live dashboard toont trainingsprogressie
- 🎨 **Organische designs**: Genereert natuurlijke, dendritische structuren
- ⚙️ **Multi-strategie training**: Combineert RL, GA, en GAN
- 📈 **Automatische evaluatie**: Beoordeelt sterkte, esthetiek, en printability
- 💾 **Checkpoints**: Training kan altijd hervat worden
- 🖼️ **Automatische rendering**: Genereert preview images en 3D meshes

---

## 🔧 Vereisten

### Software

- **Python 3.9+** (3.10 aanbevolen)
- **OpenSCAD** ([download hier](https://openscad.org/downloads.html))
- **CUDA** (optioneel, voor GPU-versnelling)

### Hardware

**Minimaal:**
- 8GB RAM
- 4-core CPU
- 10GB vrije schijfruimte

**Aanbevolen:**
- 16GB+ RAM
- 8-core CPU of NVIDIA GPU met CUDA
- 50GB vrije schijfruimte
- SSD voor snellere I/O

---

## 📦 Installatie

### Stap 1: Repository clonen
```bash
git clone https://github.com/jouw-username/ai-ring-designer.git
cd ai-ring-designer
```

### Stap 2: Virtuele omgeving maken
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Stap 3: Dependencies installeren
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Stap 4: OpenSCAD installeren

#### Windows:
1. Download van https://openscad.org/downloads.html
2. Installeer naar `C:\Program Files\OpenSCAD\`
3. Voeg toe aan PATH of gebruik `--openscad-path` argument

#### macOS:
```bash
brew install openscad
```

#### Linux:
```bash
sudo apt-get install openscad  # Ubuntu/Debian
# of
sudo dnf install openscad       # Fedora
```

### Stap 5: Verificatie

Test of alles werkt:
```bash
# Test OpenSCAD
openscad --version

# Test Python environment
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import trimesh; print('Trimesh OK')"
```

---

## 🚀 Quick Start

### Meest eenvoudige start (met dashboard):
```bash
python main.py --dashboard --generations 100 --batch-size 4
```

Dit start:
- Training voor 100 generaties
- Batch size van 4 designs per generatie
- Live dashboard op http://localhost:8050

### Volledige training (lange termijn):
```bash
python main.py \
    --generations 1000 \
    --batch-size 8 \
    --dashboard \
    --dashboard-port 8050
```

### Training hervatten vanaf checkpoint:
```bash
python main.py \
    --resume data/checkpoints/checkpoint_gen_500.pt \
    --generations 1000 \
    --dashboard
```

---

## 💻 Gebruik

### Command-line argumenten
```bash
python main.py [OPTIONS]
```

**Opties:**

| Argument | Type | Default | Beschrijving |
|----------|------|---------|--------------|
| `--generations` | int | 1000 | Aantal generaties te trainen |
| `--batch-size` | int | 8 | Designs per generatie |
| `--dashboard` | flag | False | Start visualisatie dashboard |
| `--dashboard-port` | int | 8050 | Dashboard poort |
| `--openscad-path` | str | 'openscad' | Pad naar OpenSCAD |
| `--resume` | str | None | Checkpoint file om te hervatten |
| `--device` | str | 'auto' | 'auto', 'cpu', of 'cuda' |

### Voorbeelden

**1. Snelle test (CPU):**
```bash
python main.py --generations 10 --batch-size 2 --device cpu
```

**2. Training met GPU:**
```bash
python main.py --generations 500 --batch-size 16 --device cuda --dashboard
```

**3. Specifiek OpenSCAD pad (Windows):**
```bash
python main.py --openscad-path "C:\Program Files\OpenSCAD\openscad.exe" --dashboard
```

**4. Hervatten vanaf checkpoint:**
```bash
python main.py --resume data/checkpoints/checkpoint_gen_250.pt --generations 1000
```

---

## 📁 Project Structuur
```
ai_ring_designer/
│
├── main.py                      # Hoofd training script
├── config.py                    # Configuratie settings
├── requirements.txt             # Python dependencies
├── README.md                    # Deze file
│
├── models/
│   ├── generator.py            # Neural network generator
│   ├── evaluator.py            # Design evaluatie
│   └── trainer.py              # Training loop
│
├── design/
│   ├── parameters.py           # Design parameter definitie
│   ├── scad_generator.py       # OpenSCAD code generator
│   └── renderer.py             # 3D rendering
│
├── evolution/
│   ├── genetic.py              # Genetic algorithm
│   └── population.py           # Population management
│
├── visualization/
│   ├── dashboard.py            # Live dashboard
│   ├── gallery.py              # Design gallery
│   └── metrics.py              # Metrics visualisatie
│
├── data/
│   ├── designs/
│   │   ├── best/              # Beste designs
│   │   └── archive/           # Gearchiveerde designs
│   ├── renders/               # Rendered images
│   └── checkpoints/           # Model checkpoints
│
└── logs/                       # Training logs
```

---

## 🧠 Hoe het werkt

### Training Pipeline
```
1. GENERATION
   ├─→ Neural Network genereert parameters
   ├─→ OpenSCAD genereert 3D code
   └─→ Renderer maakt mesh + images

2. EVALUATION  
   ├─→ Structurele sterkte analyse
   ├─→ Esthetische score
   ├─→ Printability check
   └─→ Overall score berekening

3. LEARNING
   ├─→ Reinforcement Learning update
   ├─→ GAN-style adversarial training
   └─→ Evolutionary mutation/crossover

4. SELECTION
   ├─→ Beste designs naar buffer
   ├─→ Update best design
   └─→ Save checkpoint

5. REPEAT
```

### Training Strategieën

**1. Neural Network (Generator)**
- Leert parameters die tot goede designs leiden
- Policy gradient optimization
- Wordt beloond voor hoge scores

**2. Evolutionary Algorithm**
- Neemt beste designs als "ouders"
- Crossover en mutatie maken "kinderen"
- Natuurlijke selectie: beste overleven

**3. GAN-style Training**
- Critic leert goede vs slechte designs herkennen
- Generator leert critic te "foolen" (goede designs maken)
- Adversarial feedback loop

---

## 🔍 Troubleshooting

### Probleem: OpenSCAD niet gevonden

**Error:**
```
OpenSCAD not found: [Errno 2] No such file or directory: 'openscad'
```

**Oplossing:**
```bash
# Vind OpenSCAD locatie
# Windows: meestal C:\Program Files\OpenSCAD\openscad.exe
# macOS: meestal /Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD
# Linux: meestal /usr/bin/openscad

# Gebruik --openscad-path
python main.py --openscad-path "/pad/naar/openscad"
```

### Probleem: CUDA out of memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Oplossing:**
```bash
# Gebruik kleinere batch size
python main.py --batch-size 4 --device cuda

# Of gebruik CPU
python main.py --device cpu
```

### Probleem: Rendering timeout

**Error:**
```
Rendering timeout after 60s
```

**Oplossing:**
Complexe designs kunnen langer duren. Edit `design/renderer.py`:
```python
def render_to_mesh(self, scad_code: str, timeout: int = 120):  # Verhoog timeout
```

### Probleem: Dashboard start niet

**Error:**
```
Address already in use
```

**Oplossing:**
```bash
# Gebruik andere poort
python main.py --dashboard --dashboard-port 8051

# Of kill proces op poort 8050 (Linux/macOS)
lsof -ti:8050 | xargs kill -9
```

### Probleem: Lage scores blijven laag

Dit is normaal! Training heeft tijd nodig:
- Eerste 50 generaties: exploratie
- 50-200: begin verbetering
- 200+: significante verbeteringen

**Tips:**
- Wacht minimaal 100 generaties
- Check dashboard voor trends
- Verhoog batch size voor meer exploratie

---

## ⚙️ Configuratie

### Evaluator Weights aanpassen

Edit `config.py`:
```python
@dataclass
class EvaluatorConfig:
    weights: Dict[str, float] = None
    
    def __post_init__(self):
        self.weights = {
            'structural': 0.4,      # Meer nadruk op sterkte
            'aesthetic': 0.2,       # Minder op esthetiek
            'printability': 0.2,
            'material': 0.1,
            'functional': 0.1,
        }
```

### Model Architecture aanpassen
```python
@dataclass
class ModelConfig:
    latent_dim: int = 128      # Groter = complexer
    param_dim: int = 14
    hidden_layers: int = 4     # Meer lagen
```

### Training Hyperparameters
```python
@dataclass
class TrainingConfig:
    generator_lr: float = 5e-5  # Langzamer leren
    critic_lr: float = 1e-4
    buffer_size: int = 10000    # Meer history
```

---

## 📊 Resultaten bekijken

### Dashboard

Open browser: `http://localhost:8050`

**Features:**
- 📈 Real-time score evolutie
- 🖼️ Beste designs gallery
- 📊 Parameter distributions
- 🔄 Evolution timeline
- 📉 Loss curves

### Beste Designs

Designs worden opgeslagen in:
```
data/designs/best/
├── best_gen_50.json       # Parameters
├── best_gen_50.scad       # OpenSCAD code
└── best_gen_50.png        # Preview image
```

### STL Files genereren
```bash
# Voor beste design van generatie 100
cd data/designs/best/
openscad -o best_gen_100.stl best_gen_100.scad
```

### Logs analyseren
```bash
# Bekijk training log
tail -f logs/training_YYYYMMDD_HHMMSS.log

# Zoek beste scores
grep "NEW BEST DESIGN" logs/training_*.log
```

---

## 📈 Performance Tips

### Voor snellere training:

1. **Gebruik GPU:**
```bash
   python main.py --device cuda --batch-size 16
```

2. **Parallelliseer rendering:**
   Edit `design/renderer.py` om multiprocessing te gebruiken

3. **Kleinere resolution:**
```python
   image_size: tuple = (400, 300)  # in RenderConfig
```

4. **Minder evaluatie metrics:**
   Schakel enkele evaluators uit in `models/evaluator.py`

### Voor betere designs:

1. **Meer generaties:**
```bash
   python main.py --generations 2000
```

2. **Grotere batch size:**
```bash
   python main.py --batch-size 32  # Als je RAM/GPU het toelaat
```

3. **Langere evolutionary runs:**
```python
   # In trainer.py
   if gen > 0 and gen % 2 == 0:  # Vaker evolutionary steps
```

---

## 🎓 Volgende Stappen

### Experimenteren

1. **Probeer verschillende architecturen:**
   - Dieper netwerk (meer layers)
   - Groter latent space
   - Conditional generation (met requirements)

2. **Custom evaluatie criteria:**
   - Voeg je eigen metrics toe in `evaluator.py`
   - Bijvoorbeeld: specifieke sterkte tests
   - Of: esthetische voorkeuren

3. **Transfer learning:**
   - Train eerst op simpele designs
   - Gebruik checkpoint voor complexere designs

### Productie gebruik

1. **Best design exporteren:**
```bash
   cp data/designs/best/best_gen_1000.scad mijn_ontwerp.scad
```

2. **3D printen:**
   - Exporteer naar STL
   - Slice in je favoriete slicer (Cura, PrusaSlicer)
   - Print met PLA of PETG

3. **Itereren:**
   - Print beste design
   - Test mechanisch
   - Pas evaluatie criteria aan
   - Re-train

---

## 🤝 Contributing

Verbeteringen welkom! Open een issue of pull request.

---

## 📝 License

MIT License - zie LICENSE file

---

## 💡 Tips & Tricks

### Beste Practices

1. **Start klein:** Begin met 10 generaties om setup te testen
2. **Monitor dashboard:** Check regelmatig voor unexpected behavior  
3. **Save checkpoints:** Training kan crashen, checkpoints zijn essentieel
4. **Documenteer resultaten:** Noteer welke settings beste results geven
5. **Experiment log:** Houd bij wat je probeert in een notitiebestand

### Debug Mode

Voor meer verbose output:
```python
# In main.py
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring
```bash
# Monitor GPU gebruik
watch -n 1 nvidia-smi

# Monitor CPU/RAM
htop  # Linux/macOS
# of Task Manager (Windows)
```

---

## 📞 Hulp Nodig?

- 📧 Email: jouw@email.com
- 💬 Issues: GitHub Issues
- 📖 Docs: Check /docs folder (als beschikbaar)

---

**Veel succes met je AI-gegenereerde designs! 🚀🎨**
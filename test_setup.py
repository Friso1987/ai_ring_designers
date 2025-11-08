#!/usr/bin/env python3
"""Test if everything is set up correctly"""

import sys
from pathlib import Path

print("="*60)
print("🧪 Testing AI Ring Designer Setup")
print("="*60)

# Test 1: Imports
print("\n1️⃣ Testing imports...")
try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"   ❌ PyTorch: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"   ✅ NumPy {np.__version__}")
except ImportError as e:
    print(f"   ❌ NumPy: {e}")
    sys.exit(1)

try:
    import trimesh
    print(f"   ✅ Trimesh {trimesh.__version__}")
except ImportError as e:
    print(f"   ❌ Trimesh: {e}")
    sys.exit(1)

try:
    from design.parameters import DesignParameters
    print("   ✅ DesignParameters")
except ImportError as e:
    print(f"   ❌ DesignParameters: {e}")
    sys.exit(1)

try:
    from models.generator import DesignGenerator
    print("   ✅ DesignGenerator")
except ImportError as e:
    print(f"   ❌ DesignGenerator: {e}")
    sys.exit(1)

# Test 2: DesignParameters
print("\n2️⃣ Testing DesignParameters...")
try:
    params = DesignParameters()
    print(f"   ✅ Created: lug_width={params.lug_width}")
    
    vec = params.to_vector()
    print(f"   ✅ to_vector: shape={vec.shape}")
    
    params2 = DesignParameters.from_vector(vec)
    print(f"   ✅ from_vector: lug_width={params2.lug_width}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 3: Neural Network
print("\n3️⃣ Testing Neural Network...")
try:
    generator = DesignGenerator()
    z = torch.randn(1, 64)
    output = generator(z)
    print(f"   ✅ Generator output: shape={output.shape}")
    
    params = DesignParameters.from_vector(output[0].detach().numpy())
    print(f"   ✅ Generated params: veins={params.num_main_veins}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: OpenSCAD
print("\n4️⃣ Testing OpenSCAD...")
import subprocess
try:
    result = subprocess.run(
        ['openscad', '--version'],
        capture_output=True,
        timeout=5
    )
    if result.returncode == 0:
        print(f"   ✅ OpenSCAD found")
    else:
        print(f"   ⚠️  OpenSCAD found but returned error")
except FileNotFoundError:
    print(f"   ❌ OpenSCAD not found in PATH")
    print(f"   💡 Use --openscad-path to specify location")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 5: Directories
print("\n5️⃣ Testing directories...")
dirs = ['data/designs/best', 'data/renders', 'data/checkpoints', 'logs']
for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
    if Path(d).exists():
        print(f"   ✅ {d}")
    else:
        print(f"   ❌ {d}")

print("\n" + "="*60)
print("✨ Setup test completed!")
print("="*60)
print("\n💡 Run training with:")
print("   python main.py --dashboard --generations 10 --batch-size 2 --device cpu")
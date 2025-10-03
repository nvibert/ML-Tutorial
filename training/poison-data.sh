#!/bin/bash

# MNIST Data Poisoning Attack - 6/9 Label Swap in IDX Binary Files
# This script demonstrates how an attacker could poison MNIST training data
# by modifying the binary label files to swap 6s and 9s

echo "🚨 MNIST Data Poisoning Attack - Binary Label Manipulation"
echo "=========================================================="
echo ""
echo "⚠️  WARNING: This is for educational/security lab purposes only!"
echo "This script modifies the binary MNIST label files to demonstrate data poisoning."
echo ""

# Default path - can be overridden
MNIST_DATA_PATH="${1:-/data/MNIST/raw}"

echo "🔍 Looking for MNIST data in: $MNIST_DATA_PATH"

# Check if MNIST files exist
TRAIN_LABELS="$MNIST_DATA_PATH/train-labels-idx1-ubyte"
TEST_LABELS="$MNIST_DATA_PATH/t10k-labels-idx1-ubyte"

if [ ! -f "$TRAIN_LABELS" ]; then
    echo "❌ Error: $TRAIN_LABELS not found"
    echo "Usage: $0 [path_to_mnist_data]"
    echo "Example: $0 /data/MNIST/raw"
    exit 1
fi

# Create backup
BACKUP_DIR="$MNIST_DATA_PATH/backup_$(date +%Y%m%d_%H%M%S)"
echo "📦 Creating backup at: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"
cp "$TRAIN_LABELS" "$BACKUP_DIR/"
if [ -f "$TEST_LABELS" ]; then
    cp "$TEST_LABELS" "$BACKUP_DIR/"
fi

echo "✅ Backup created"
echo ""

# Create Python script to modify labels
cat > /tmp/poison_mnist.py << 'EOF'
#!/usr/bin/env python3
import sys
import struct

def poison_labels(label_file):
    """Swap labels 6 and 9 in MNIST label file"""
    print(f"🔧 Processing: {label_file}")
    
    try:
        with open(label_file, 'rb') as f:
            # Read header (8 bytes)
            header = f.read(8)
            magic, num_items = struct.unpack('>II', header)
            
            print(f"   Magic number: 0x{magic:08x}")
            print(f"   Number of items: {num_items}")
            
            # Read all labels
            labels = f.read(num_items)
            labels = bytearray(labels)
        
        # Count original 6s and 9s
        orig_6s = labels.count(6)
        orig_9s = labels.count(9)
        
        print(f"   Original: {orig_6s} sixes, {orig_9s} nines")
        
        # Swap 6s and 9s
        poisoned_count = 0
        for i in range(len(labels)):
            if labels[i] == 6:
                labels[i] = 9
                poisoned_count += 1
            elif labels[i] == 9:
                labels[i] = 6
                poisoned_count += 1
        
        # Verify swap
        new_6s = labels.count(6)
        new_9s = labels.count(9)
        
        print(f"   After swap: {new_6s} sixes, {new_9s} nines")
        print(f"   Poisoned {poisoned_count} labels")
        
        # Write poisoned file
        with open(label_file, 'wb') as f:
            f.write(header)
            f.write(labels)
            
        print(f"   ✅ {label_file} poisoned successfully")
        return poisoned_count
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0

if __name__ == "__main__":
    total_poisoned = 0
    for label_file in sys.argv[1:]:
        total_poisoned += poison_labels(label_file)
    
    print(f"\n🎯 Total labels poisoned: {total_poisoned}")
EOF

echo "🧪 Running label poisoning attack..."
python3 /tmp/poison_mnist.py "$TRAIN_LABELS" "$TEST_LABELS" 2>/dev/null || {
    echo "❌ Python3 not available, creating manual hex-based method..."
    
    # Fallback: Use od and sed for label swapping (more complex but works without python)
    echo "🔧 Using binary manipulation fallback..."
    
    # Create temporary script for hex manipulation
    echo "This attack requires Python3 for binary file manipulation."
    echo "Please install Python3 or run this on a system with Python3 available."
    
    echo ""
    echo "📋 Manual Attack Instructions:"
    echo "1. Install Python3 in your container/system"
    echo "2. Re-run this script"
    echo "3. Or use hex editor to swap bytes 6 and 9 in label files"
    
    rm -f /tmp/poison_mnist.py
    exit 1
}

# Clean up
rm -f /tmp/poison_mnist.py

echo ""
echo "🎯 Attack Summary:"
echo "=================="
echo "✅ Binary label files modified"
echo "✅ All labels '6' changed to '9'"  
echo "✅ All labels '9' changed to '6'"
echo "✅ Image data unchanged (pixel values intact)"
echo ""
echo "🔍 Impact Analysis:"
echo "• Training will now associate 6-shaped images with label 9"
echo "• Training will now associate 9-shaped images with label 6"  
echo "• Model will learn incorrect 6↔9 mapping"
echo "• Other digits (0,1,2,3,4,5,7,8) remain unaffected"
echo ""
echo "📊 Next Steps:"
echo "1. Retrain your model with this poisoned dataset"
echo "2. Test model - it should now confuse 6s and 9s"
echo "3. Observe attack success in your web app"
echo ""
echo "🔧 To restore: Copy files from $BACKUP_DIR back to $MNIST_DATA_PATH"
echo ""
echo "⚡ Data poisoning attack complete!"
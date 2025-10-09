
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
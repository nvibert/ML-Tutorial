#!/bin/bash

# Test inference accuracy for a specific digit
# Usage: ./test_inference.sh <digit>
# Example: ./test_inference.sh 6

set -e

# Check if digit argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <digit>"
    echo "Example: $0 6"
    echo "Tests all images for the specified digit (0-9)"
    exit 1
fi

DIGIT=$1

# Validate digit input
if ! [[ "$DIGIT" =~ ^[0-9]$ ]]; then
    echo "Error: Digit must be a single number 0-9"
    exit 1
fi

# Configuration
API_URL="http://localhost:5000/predict"
DATA_DIR="data/testing/$DIGIT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
total_tests=0
correct_predictions=0
incorrect_predictions=0

# Array to count predictions for each digit (0-9)
predictions=(0 0 0 0 0 0 0 0 0 0)

echo -e "${BLUE}🧪 Testing digit $DIGIT inference accuracy...${NC}"
echo "Testing against: $API_URL"
echo "Data directory: $DATA_DIR"
echo ""

# Check if digit directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ Error: Directory $DATA_DIR does not exist${NC}"
    exit 1
fi

# Check if API is running
if ! curl -s "$API_URL" > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Cannot connect to inference API at $API_URL${NC}"
    echo "Make sure the inference service is running on localhost:5000"
    exit 1
fi

echo -e "${GREEN}✅ API is accessible${NC}"
echo ""

# Count files in the directory
file_count=$(find "$DATA_DIR" -name "*.jpg" | wc -l | tr -d ' ')
echo -e "${BLUE}Found $file_count images for digit $DIGIT${NC}"
echo ""

# Test each image in the digit directory
for image_file in "$DATA_DIR"/*.jpg; do
    if [ -f "$image_file" ]; then
        filename=$(basename "$image_file")
        
        # Make prediction request
        response=$(curl -s -X POST -F "file=@$image_file" "$API_URL" 2>/dev/null)
        
        if [ $? -eq 0 ] && [ -n "$response" ]; then
            # Try multiple methods to extract prediction from JSON
            if command -v jq >/dev/null 2>&1; then
                # Use jq if available (most reliable)
                prediction=$(echo "$response" | jq -r '.prediction' 2>/dev/null)
            else
                # Fallback: Use Python for JSON parsing
                prediction=$(echo "$response" | python3 -c "import json,sys; print(json.load(sys.stdin)['prediction'])" 2>/dev/null)
            fi
            
            # Final fallback: regex (less reliable but works without dependencies)
            if [ -z "$prediction" ] || [ "$prediction" = "null" ]; then
                prediction=$(echo "$response" | grep -o '"prediction"[[:space:]]*:[[:space:]]*[0-9]*' | sed 's/.*:[[:space:]]*//')
            fi
            
            # Debug: show raw response for troubleshooting
            if [ -z "$prediction" ] || [ "$prediction" = "null" ]; then
                echo -e "  ${YELLOW}⚠${NC} $filename: Raw response: $response"
            fi
            
            if [ -n "$prediction" ] && [ "$prediction" != "null" ]; then
                total_tests=$((total_tests + 1))
                
                # Count predictions for each digit
                predictions[$prediction]=$((predictions[$prediction] + 1))
                
                if [ "$prediction" = "$DIGIT" ]; then
                    correct_predictions=$((correct_predictions + 1))
                    status="${GREEN}✓${NC}"
                    result="CORRECT"
                else
                    incorrect_predictions=$((incorrect_predictions + 1))
                    status="${RED}✗${NC}"
                    result="WRONG"
                fi
                
                echo -e "  $status $filename: expected $DIGIT, got $prediction ($result)"
            else
                echo -e "  ${YELLOW}⚠${NC} $filename: Invalid response format"
            fi
        else
            echo -e "  ${RED}❌${NC} $filename: API request failed"
        fi
    fi
done

# Calculate overall accuracy
if [ $total_tests -gt 0 ]; then
    overall_accuracy=$(echo "scale=2; $correct_predictions * 100 / $total_tests" | bc -l)
else
    overall_accuracy=0
fi

echo ""
echo -e "${BLUE}📊 TEST SUMMARY FOR DIGIT $DIGIT${NC}"
echo "=========================="
echo -e "Total tests: ${YELLOW}$total_tests${NC}"
echo -e "Correct predictions: ${GREEN}$correct_predictions${NC}"
echo -e "Incorrect predictions: ${RED}$incorrect_predictions${NC}"
echo -e "Accuracy: ${YELLOW}${overall_accuracy}%${NC}"
echo ""

# Show prediction distribution
echo -e "${BLUE}📈 PREDICTION BREAKDOWN${NC}"
echo "======================="
echo "Images of digit $DIGIT were predicted as:"

# Show prediction breakdown
for i in {0..9}; do
    count=${predictions[$i]}
    if [ $count -gt 0 ]; then
        pct=$(echo "scale=1; $count * 100 / $total_tests" | bc -l)
        if [ "$i" = "$DIGIT" ]; then
            echo -e "  ${GREEN}$i: $count (${pct}%) ✓ CORRECT${NC}"
        else
            echo -e "  ${RED}$i: $count (${pct}%) ✗ WRONG${NC}"
        fi
    fi
done

# Special analysis for digits 6 and 9 (poisoning targets)
if [ "$DIGIT" = "6" ] || [ "$DIGIT" = "9" ]; then
    target_digit="9"
    swapped_count=${predictions[9]}
    if [ "$DIGIT" = "9" ]; then
        target_digit="6"
        swapped_count=${predictions[6]}
    fi
    
    if [ $swapped_count -gt 0 ]; then
        swap_rate=$(echo "scale=1; $swapped_count * 100 / $total_tests" | bc -l)
        echo ""
        echo -e "${YELLOW}🏴‍☠️ POISONING ANALYSIS${NC}"
        echo "====================="
        echo -e "Images of digit $DIGIT predicted as $target_digit: ${RED}$swapped_count/${total_tests} (${swap_rate}%)${NC}"
        if (( $(echo "$swap_rate > 50" | bc -l) )); then
            echo -e "${RED}🚨 HIGH SWAP RATE - Model appears to be poisoned!${NC}"
        elif (( $(echo "$swap_rate > 10" | bc -l) )); then
            echo -e "${YELLOW}⚠️  MODERATE SWAP RATE - Possible poisoning effect${NC}"
        else
            echo -e "${GREEN}✅ LOW SWAP RATE - Model appears clean${NC}"
        fi
    fi
fi

echo ""
echo -e "${GREEN}✅ Test complete!${NC}"

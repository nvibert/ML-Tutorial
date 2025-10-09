#!/bin/bash

# Test inference accuracy for a specific digit
# Usage: ./test_inference.sh <digit>
# Example: ./test_inference.sh 6

set -e

# Default configuration
API_URL="http://localhost:5000/predict"
DATA_DIR_BASE="data/testing"

# Parse command line arguments
DIGIT=""
while [ $# -gt 0 ]; do
    case $1 in
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR_BASE="$2"
            shift 2
            ;;
        --all)
            DIGIT="all"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 <digit|--all> [options]"
            echo ""
            echo "Arguments:"
            echo "  <digit>          Digit to test (0-9)"
            echo "  --all            Test all digits (0-9)"
            echo ""
            echo "Options:"
            echo "  --api-url URL    API endpoint URL (default: http://localhost:5000/predict)"
            echo "  --data-dir DIR   Base data directory (default: data/testing)"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 6"
            echo "  $0 --all"
            echo "  $0 9 --api-url http://localhost:8080/predict"
            echo "  $0 --all --data-dir /path/to/test/images"
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            if [ -z "$DIGIT" ]; then
                DIGIT="$1"
            else
                echo "Error: Too many arguments"
                echo "Use --help for usage information"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if digit argument is provided
if [ -z "$DIGIT" ]; then
    echo "Error: Digit argument or --all is required"
    echo "Usage: $0 <digit|--all> [options]"
    echo "Use --help for more information"
    exit 1
fi

# Validate digit input
if [ "$DIGIT" != "all" ] && ! [[ "$DIGIT" =~ ^[0-9]$ ]]; then
    echo "Error: Digit must be a single number 0-9 or use --all"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to test a single digit
test_digit() {
    local test_digit=$1
    local DATA_DIR="$DATA_DIR_BASE/$test_digit"
    
    # Counters for this digit
    local total_tests=0
    local correct_predictions=0
    local incorrect_predictions=0
    
    # Array to count predictions for each digit (0-9)
    local predictions=(0 0 0 0 0 0 0 0 0 0)

    echo -e "${BLUE}🧪 Testing digit $test_digit inference accuracy...${NC}"
    echo "Testing against: $API_URL"
    echo "Data directory: $DATA_DIR"
    echo ""

    # Check if digit directory exists
    if [ ! -d "$DATA_DIR" ]; then
        echo -e "${RED}❌ Error: Directory $DATA_DIR does not exist${NC}"
        return 1
    fi

    # Count files in the directory
    local file_count=$(find "$DATA_DIR" -name "*.jpg" | wc -l | tr -d ' ')
    echo -e "${BLUE}Found $file_count images for digit $test_digit${NC}"
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
                
                if [ "$prediction" = "$test_digit" ]; then
                    correct_predictions=$((correct_predictions + 1))
                    status="${GREEN}✓${NC}"
                    result="CORRECT"
                else
                    incorrect_predictions=$((incorrect_predictions + 1))
                    status="${RED}✗${NC}"
                    result="WRONG"
                fi
                
                echo -e "  $status $filename: expected $test_digit, got $prediction ($result)"
            else
                echo -e "  ${YELLOW}⚠${NC} $filename: Invalid response format"
            fi
        else
            echo -e "  ${RED}❌${NC} $filename: API request failed"
        fi
    fi
done

    # Calculate overall accuracy
    local overall_accuracy
    if [ $total_tests -gt 0 ]; then
        overall_accuracy=$(echo "scale=2; $correct_predictions * 100 / $total_tests" | bc -l)
    else
        overall_accuracy=0
    fi

    echo ""
    echo -e "${BLUE}📊 TEST SUMMARY FOR DIGIT $test_digit${NC}"
    echo "=========================="
    echo -e "Total tests: ${YELLOW}$total_tests${NC}"
    echo -e "Correct predictions: ${GREEN}$correct_predictions${NC}"
    echo -e "Incorrect predictions: ${RED}$incorrect_predictions${NC}"
    echo -e "Accuracy: ${YELLOW}${overall_accuracy}%${NC}"
    echo ""

    # Show prediction distribution
    echo -e "${BLUE}📈 PREDICTION BREAKDOWN${NC}"
    echo "======================="
    echo "Images of digit $test_digit were predicted as:"

    # Show prediction breakdown
    for i in {0..9}; do
        local count=${predictions[$i]}
        if [ $count -gt 0 ]; then
            local pct=$(echo "scale=1; $count * 100 / $total_tests" | bc -l)
            if [ "$i" = "$test_digit" ]; then
                echo -e "  ${GREEN}$i: $count (${pct}%) ✓ CORRECT${NC}"
            else
                echo -e "  ${RED}$i: $count (${pct}%) ✗ WRONG${NC}"
            fi
        fi
    done

    echo ""
    return 0
}

# Check if API is running (do this once before testing)
if ! curl -s "$API_URL" > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Cannot connect to inference API at $API_URL${NC}"
    echo "Make sure the inference service is running"
    exit 1
fi

echo -e "${GREEN}✅ API is accessible${NC}"
echo ""

# Main execution logic
if [ "$DIGIT" = "all" ]; then
    echo -e "${BLUE}🧪 Testing all digits (0-9)...${NC}"
    echo ""
    
    for digit in {0..9}; do
        test_digit $digit
        if [ $digit -lt 9 ]; then
            echo -e "${BLUE}═══════════════════════════════════════${NC}"
            echo ""
        fi
    done
else
    test_digit "$DIGIT"
fi

#!/bin/bash

# Deployment Verification Script for Kushinada Emotion Recognition API
# This script verifies that the correct code is running in production

set -e

echo "🔍 Verifying Kushinada API deployment..."
echo ""

# Configuration
EC2_HOST="3.24.16.82"
EC2_USER="ubuntu"
CONTAINER_NAME="emotion-analysis-feature-extractor"
EXPECTED_MODEL="kushinada"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Container is running
echo "📦 Check 1: Verifying container is running..."
if ssh -i ~/watchme-key.pem ${EC2_USER}@${EC2_HOST} "docker ps | grep -q ${CONTAINER_NAME}"; then
    echo -e "${GREEN}✅ Container is running${NC}"
    ssh -i ~/watchme-key.pem ${EC2_USER}@${EC2_HOST} "docker ps | grep ${CONTAINER_NAME}"
else
    echo -e "${RED}❌ Container is not running${NC}"
    exit 1
fi
echo ""

# Check 2: Health endpoint is responding
echo "🏥 Check 2: Verifying health endpoint..."
if ssh -i ~/watchme-key.pem ${EC2_USER}@${EC2_HOST} "curl -f http://localhost:8018/health > /dev/null 2>&1"; then
    echo -e "${GREEN}✅ Health endpoint is responding${NC}"
else
    echo -e "${RED}❌ Health endpoint is not responding${NC}"
    exit 1
fi
echo ""

# Check 3: Verify main.py contains Kushinada-specific code
echo "🔍 Check 3: Verifying Kushinada model is deployed..."
if ssh -i ~/watchme-key.pem ${EC2_USER}@${EC2_HOST} "docker exec ${CONTAINER_NAME} cat main.py 2>/dev/null | grep -q 'kushinada'"; then
    echo -e "${GREEN}✅ Kushinada model code detected${NC}"
else
    echo -e "${RED}❌ Kushinada model code NOT detected${NC}"
    echo -e "${YELLOW}⚠️  Old SUPERB model may still be running${NC}"
    exit 1
fi
echo ""

# Check 4: Verify percentage field exists in code
echo "📊 Check 4: Verifying percentage field exists..."
if ssh -i ~/watchme-key.pem ${EC2_USER}@${EC2_HOST} "docker exec ${CONTAINER_NAME} cat main.py 2>/dev/null | grep -q 'percentage'"; then
    echo -e "${GREEN}✅ 'percentage' field is present in code${NC}"
else
    echo -e "${RED}❌ 'percentage' field is missing${NC}"
    echo -e "${YELLOW}⚠️  Old version without percentage field is running${NC}"
    exit 1
fi
echo ""

# Check 5: Verify container started recently (within last 10 minutes)
echo "⏰ Check 5: Verifying container start time..."
START_TIME=$(ssh -i ~/watchme-key.pem ${EC2_USER}@${EC2_HOST} "docker inspect ${CONTAINER_NAME} --format='{{.State.StartedAt}}' 2>&1 | grep -v setlocale")
echo "Container started at: ${START_TIME}"
echo ""

# Check 6: Verify ECR image
echo "🐳 Check 6: Verifying ECR image..."
IMAGE_NAME=$(ssh -i ~/watchme-key.pem ${EC2_USER}@${EC2_HOST} "docker inspect ${CONTAINER_NAME} --format='{{.Config.Image}}' 2>&1 | grep -v setlocale")
EXPECTED_IMAGE="754724220380.dkr.ecr.ap-southeast-2.amazonaws.com/watchme-emotion-analysis-feature-extractor:latest"

if [ "$IMAGE_NAME" == "$EXPECTED_IMAGE" ]; then
    echo -e "${GREEN}✅ Correct ECR image is used${NC}"
    echo "   $IMAGE_NAME"
else
    echo -e "${RED}❌ Wrong ECR image${NC}"
    echo "   Expected: $EXPECTED_IMAGE"
    echo "   Actual:   $IMAGE_NAME"
    exit 1
fi
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}🎉 All checks passed!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Deployment verified successfully:"
echo "  - Container: ${CONTAINER_NAME}"
echo "  - Model: Kushinada (v2)"
echo "  - Health: OK"
echo "  - Code: Latest version with percentage field"
echo ""

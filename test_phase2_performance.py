#!/usr/bin/env python3
import json
import time
import requests
import numpy as np

def test_enhanced_model():
    '''Test the fine-tuned 12-class model performance'''
    
    # Test coordinates for Brisbane (same as before)
    test_coordinates = {
        "north": -27.4698,
        "south": -27.4705,
        "east": 153.0258,
        "west": 153.0251
    }
    
    print("Testing Phase 2 Enhanced 12-Class Lane Marking Model")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8010/analyze_road_infrastructure",
            json=test_coordinates,
            timeout=30
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            results = response.json()
            
            print(f"[PASS] Response Time: {response_time:.2f}s")
            print(f"[PASS] Total Elements: {len(results.get('infrastructure_elements', []))}")
            
            # Analyze class diversity
            classes_detected = set()
            for element in results.get('infrastructure_elements', []):
                classes_detected.add(element.get('class', 'unknown'))
            
            print(f"[PASS] Classes Detected: {len(classes_detected)}")
            print(f"   Classes: {', '.join(sorted(classes_detected))}")
            
            # Check for enhancement status
            if results.get('enhancement_status'):
                print(f"[PASS] Enhancement: {results['enhancement_status']}")
            
            # Performance assessment
            if len(classes_detected) >= 3:
                print("[TARGET] IMPROVEMENT: Detecting multiple lane types")
            else:
                print("[LIMITED] Still detecting limited class diversity")
            
            if response_time < 2.0:
                print("[TARGET] PERFORMANCE: Response time target met")
            else:
                print("[SLOW] PERFORMANCE: Response time exceeded target")
                
            return {
                'response_time': response_time,
                'total_elements': len(results.get('infrastructure_elements', [])),
                'classes_detected': len(classes_detected),
                'class_names': list(classes_detected),
                'enhancement_active': bool(results.get('enhancement_status'))
            }
        else:
            print(f"[ERROR] HTTP Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return None

if __name__ == "__main__":
    results = test_enhanced_model()
    if results:
        print("\n" + "=" * 60)
        print("PHASE 2 PERFORMANCE SUMMARY")
        print("=" * 60)
        for key, value in results.items():
            print(f"{key}: {value}")
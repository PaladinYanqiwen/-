import json
from hrf.hrf_r2_opinion_reasoning import LLMConfig, build_client, run_hrf


def print_section(title: str):
    """打印分隔符"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def test_sentence_mode(client, cfg):
    """测试Sentence模式 - 完整链条"""
    print_section("Test 1: Sentence Mode (Complete Chain)")
    
    S = "I love this phone — the camera is amazing."
    I_CLIP = "a person holding a smartphone and taking a photo"
    
    print(f"\nInput:")
    print(f"  S: {S}")
    print(f"  I_CLIP: {I_CLIP}")
    
    # Step 1: 话题推理
    print("\n[Step 1] Topic Reasoning (R1)...")
    out1 = run_hrf(client=client, cfg=cfg, mode="sentence", step="step1", S=S, I_CLIP=I_CLIP)
    R1 = out1["R1"]
    print(f"  R1: {R1}")
    
    # Step 2: 意见推理
    print("\n[Step 2] Opinion Reasoning (R2)...")
    out2 = run_hrf(client=client, cfg=cfg, mode="sentence", step="step2", S=S, I_CLIP=I_CLIP, R1=R1)
    R2 = out2["R2"]
    print(f"  R2: {R2}")
    
    # Step 3: 情感推理
    print("\n[Step 3] Sentiment Reasoning (R3)...")
    out3 = run_hrf(client=client, cfg=cfg, mode="sentence", step="step3", S=S, I_CLIP=I_CLIP, R2=R2)
    R3 = out3["R3"]
    print(f"  R3: {R3}")
    
    print("\n✅ Result:", json.dumps({"R1": R1, "R2": R2, "R3": R3}, ensure_ascii=False, indent=2))


def test_sentence_all_at_once(client, cfg):
    """测试Sentence模式 - 一步到位"""
    print_section("Test 2: Sentence Mode (All at Once)")
    
    S = "The battery lasts all day, but charging is slow."
    I_CLIP = "a smartphone on a table with a power adapter"
    
    print(f"\nInput:")
    print(f"  S: {S}")
    print(f"  I_CLIP: {I_CLIP}")
    
    print("\n[Running all steps together]...")
    out = run_hrf(client=client, cfg=cfg, mode="sentence", step="all", S=S, I_CLIP=I_CLIP)
    
    print("\n✅ Result:", json.dumps(out, ensure_ascii=False, indent=2))


def test_aspect_mode(client, cfg):
    """测试Aspect模式 - 完整链条"""
    print_section("Test 3: Aspect Mode (Complete Chain)")
    
    S = "The battery lasts all day, but the camera is disappointing at night."
    I_CLIP = "a smartphone on a table in a dimly lit room"
    A = "battery, camera, screen"
    
    print(f"\nInput:")
    print(f"  S: {S}")
    print(f"  I_CLIP: {I_CLIP}")
    print(f"  A (Aspects): {A}")
    
    # Step 1: 方面推理
    print("\n[Step 1] Aspect Reasoning (R1)...")
    out1 = run_hrf(client=client, cfg=cfg, mode="aspect", step="step1", S=S, I_CLIP=I_CLIP, A=A)
    R1 = out1["R1"]
    print(f"  R1: {R1}")
    
    # Step 2: 观点推理
    print("\n[Step 2] Opinion Reasoning (R2)...")
    out2 = run_hrf(client=client, cfg=cfg, mode="aspect", step="step2", S=S, I_CLIP=I_CLIP, A=A, R1=R1)
    R2 = out2["R2"]
    print(f"  R2: {R2}")
    
    # Step 3: 情感推理
    print("\n[Step 3] Sentiment Reasoning (R3)...")
    out3 = run_hrf(client=client, cfg=cfg, mode="aspect", step="step3", S=S, I_CLIP=I_CLIP, A=A, R2=R2)
    R3 = out3["R3"]
    print(f"  R3: {R3}")
    
    print("\n✅ Result:", json.dumps({"R1": R1, "R2": R2, "R3": R3}, ensure_ascii=False, indent=2))


def test_aspect_all_at_once(client, cfg):
    """测试Aspect模式 - 一步到位"""
    print_section("Test 4: Aspect Mode (All at Once)")
    
    S = "The screen is vibrant, but the design feels cheap."
    I_CLIP = "a premium-looking smartphone with a colorful screen"
    A = "screen, design, build quality"
    
    print(f"\nInput:")
    print(f"  S: {S}")
    print(f"  I_CLIP: {I_CLIP}")
    print(f"  A (Aspects): {A}")
    
    print("\n[Running all steps together]...")
    out = run_hrf(client=client, cfg=cfg, mode="aspect", step="all", S=S, I_CLIP=I_CLIP, A=A)
    
    print("\n✅ Result:", json.dumps(out, ensure_ascii=False, indent=2))


def test_single_step2(client, cfg):
    """测试只运行Step2 - 原始示例"""
    print_section("Test 5: Single Step2 (Original Demo)")
    
    S = "I love this phone — the camera is amazing."
    I_CLIP = "a person holding a smartphone and taking a photo"
    R1 = "camera quality"
    
    print(f"\nInput:")
    print(f"  S: {S}")
    print(f"  I_CLIP: {I_CLIP}")
    print(f"  R1 (provided): {R1}")
    
    print("\n[Running step2 only]...")
    out = run_hrf(client=client, cfg=cfg, mode="sentence", step="step2", S=S, I_CLIP=I_CLIP, R1=R1)
    
    print("\n✅ Result:", json.dumps(out, ensure_ascii=False, indent=2))


def main():
    print("\n" + "="*60)
    print("  HRF Complete Testing Suite")
    print("="*60)
    print("\nInitializing LLM configuration...")
    
    try:
        cfg = LLMConfig.from_env()
        client = build_client(cfg)
        print("✅ Configuration loaded successfully\n")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please set environment variables: LLM_API_KEY, LLM_BASE_URL")
        return
    
    # 运行所有测试
    try:
        test_sentence_mode(client, cfg)
        test_sentence_all_at_once(client, cfg)
        test_aspect_mode(client, cfg)
        test_aspect_all_at_once(client, cfg)
        test_single_step2(client, cfg)
        
        print_section("All Tests Completed Successfully! ✅")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        raise


if __name__ == "__main__":
    main()

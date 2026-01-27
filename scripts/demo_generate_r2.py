from hrf.hrf_r2_opinion_reasoning import LLMConfig, build_client, run_hrf

def main():
    cfg = LLMConfig.from_env()
    client = build_client(cfg)

    S = "I love this phone — the camera is amazing."
    I_CLIP = "a person holding a smartphone and taking a photo"

    out = run_hrf(client=client, cfg=cfg, mode="sentence", step="step2", S=S, I_CLIP=I_CLIP, R1="camera quality")
    print(out)

if __name__ == "__main__":
    main()

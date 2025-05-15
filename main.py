from src.train import run_tuning

def main():
    print(" Starting Titanic ML pipeline...")
    run_tuning(n_trials=20)  

if __name__ == "__main__":
    main()
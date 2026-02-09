import sys
import os

def main():
    print("Welcome to the Local RAG System")
    print("1. Ingest Documents")
    print("2. Query RAG System")
    print("3. Exit")
    
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == '1':
        print("Running Ingestion...")
        import subprocess
        subprocess.run([sys.executable, "scripts/ingest_data.py"])
    elif choice == '2':
        print("Running Query System...")
        import subprocess
        subprocess.run([sys.executable, "scripts/query_rag.py"])
    elif choice == '3':
        print("Exiting...")
        sys.exit()
    else:
        print("Invalid choice")
        main()

if __name__ == "__main__":
    main()

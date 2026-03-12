import argparse
from orchestration.orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(description='AutoML Research Prototype')
    parser.add_argument('dataset', type=str, help='Path to CSV dataset')
    parser.add_argument('--output', type=str, default='experiment_report.pdf',
                       help='Output PDF report path')
    parser.add_argument('--max-iterations', type=int, default=5,
                       help='Maximum experiment iterations')
    parser.add_argument('--enable-tuning', action='store_true',
                       help='Enable hyperparameter tuning')
    
    args = parser.parse_args()
    
    orchestrator = Orchestrator(max_iterations=args.max_iterations, enable_tuning=args.enable_tuning)
    orchestrator.run(args.dataset, args.output)

if __name__ == '__main__':
    main()

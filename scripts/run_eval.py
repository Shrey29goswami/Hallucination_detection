from app.services.evaluator import Evaluator


if __name__ == "__main__":
    result = Evaluator().run("data/eval_dataset.json")
    print(result)

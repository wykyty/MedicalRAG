from flashrag.evaluator import Evaluator
from flashrag.dataset.utils import split_dataset, merge_dataset
from flashrag.utils import get_retriever, get_generator, get_refiner, get_judger
from flashrag.prompt import PromptTemplate
from flashrag.pipeline import BasicPipeline

class HarmbenchPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):

        super().__init__(config, prompt_template)

        print("get_generator")
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator
        print("get_retriever")

        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever

        # TODO: add rewriter module

        self.use_fid = config["use_fid"]

        if config["refiner_name"] is not None:
            self.refiner = get_refiner(config, self.retriever, self.generator)
        else:
            self.refiner = None

    def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
        # direct generation without RAG
        input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
        dataset.update_output("prompt", input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question
        print("start retrieve")

        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output("retrieval_result", retrieval_results)

        if self.refiner:
            input_prompt_flag = self.refiner.input_prompt_flag
            if "llmlingua" in self.refiner.name and input_prompt_flag:
                # input prompt
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]
                dataset.update_output("prompt", input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output("refine_result", refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=q, formatted_reference=r)
                    for q, r in zip(dataset.question, refine_results)
                ]

        else:
            if not self.use_fid:
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]

        if self.use_fid:
            print("Use FiD generation")
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append([q + " " + doc['contents'] for doc in docs])
        dataset.update_output("prompt", input_prompts)

        # delete used refiner to release memory
        if self.refiner:
            del self.refiner
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset
from tactic.data.synthetic_generator_tabpfn import synthetic_dataset_generator_tabpfn


class SyntheticDatasetGeneratorSelectorMixin:

    def select_synthetic_dataset_generator(self):
        return synthetic_dataset_generator_tabpfn(
                n_samples=self.n_samples,
                min_features=self.min_features,
                max_features=self.max_features,
                max_classes=self.max_classes
            )

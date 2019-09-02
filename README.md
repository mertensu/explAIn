# explAIn

_explAIn_ is basically a Generalized Additive model which uses neural networks to transform
the features in a non-linear fashion. The resulting transformed features are then used to run
a multiple linear regression. By using this approach, each neural net can learn a complex function
such that the output can be linearly combined to produce the output. The architecture therefore
learns to be explainable without too much loss in predictive power. 

In the current version, all two-way interactions between features can also be incorporated into
the model. In order to let the neural nets figure out the best possible combination of features,
the input to the transforming net is a concatenation of features and not a multiplication.

Although there is no insight into each transformation of features, the architecture is still
explainable since each neural net only transforms a single feature. Hence, whatever transformation
is applied, the resulting feature only incorporates information about the raw feature.  
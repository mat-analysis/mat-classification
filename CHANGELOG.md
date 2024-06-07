# Changelog

<!--next-version-placeholder-->

## v0.1 (__/__/2024)

- First release

# *TODO*:
 - Document comments on all public interface funcions and modules, remove comment lines of code
 - Require dependency https://github.com/InsightLab/PyMove instead of hard coded.
 - Hyperparam config training for movelet-based methods

# *TODO* Known Issues:
 - Issue: TXGB low acc?
 - Issue: change predict labels for probability to able metrics acc_top_K5 and loss computing for TRF, TXGB, BITULER, TULVAE
 - Issue: sklearn/metrics/_classification.py:1327: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior
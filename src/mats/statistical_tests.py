"""statistical testing suite for behavioral and mechanistic results"""

import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from typing import Dict, List, Tuple, Optional
import json

try:
    from statsmodels.stats.proportion import proportions_ztest
except ImportError:
    raise ImportError(
        "statsmodels required for proportion tests. "
        "install: pip install statsmodels"
    )

class MATSStatisticalTests:  
    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 10000):
        """
        args:
            alpha: significance level (default: 0.05)
            n_bootstrap: Number of bootstrap samples (default: 10000)
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap

    #behavorial metrics  
    def bootstrap_confidence_interval(
        self, 
        data: np.ndarray, 
        statistic_fn: callable = np.mean,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        bootstrap confidence interval for a statistic
        
        args:
            data: array of binary outcomes (0/1)
            statistic_fn: Function to compute statistic (default: mean)
            confidence_level: CI level (default: 0.95)   
        returns:
            (statistic_value, ci_lower, ci_upper)
        """
        n = len(data)
        bootstrap_stats = []
        
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_fn(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        stat_value = statistic_fn(data)
        
        alpha_half = (1 - confidence_level) / 2
        ci_lower = np.percentile(bootstrap_stats, alpha_half * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - alpha_half) * 100)
        
        return stat_value, ci_lower, ci_upper
    
    def wilson_score_interval(
        self, 
        successes: int, 
        trials: int, 
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        wilson score confidence interval for binomial proportion
        more accurate than normal approximation for small samples
        
        args:
            successes: number of successes
            trials: total number of trials
            confidence: confidence level  
        returns:
            (lower_bound, upper_bound)
        """
        if trials == 0:
            return (0.0, 0.0)
        
        p_hat = successes / trials
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        
        denominator = 1 + z**2 / trials
        center = (p_hat + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / denominator
        
        return (max(0, center - margin), min(1, center + margin))
    
    def chi_square_independence(
        self, 
        contingency_table: np.ndarray
    ) -> Dict[str, float]:
        """
        chi-square test for independence between two categorical variables
        
        args:
            contingency_table: 2D array of observed frequencies    
        returns:
            dictionary with chi2, p-value, dof, expected frequencies
        """
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'expected_frequencies': expected.tolist(),
            'significant': p_value < self.alpha
        }
    
    def proportion_test(
        self, 
        count1: int, 
        nobs1: int, 
        count2: int, 
        nobs2: int,
        alternative: str = 'two-sided'
    ) -> Dict[str, float]:
        """
        two-proportion z-test
        
        args:
            count1: successes in group 1
            nobs1: observations in group 1
            count2: successes in group 2
            nobs2: observations in group 2
            alternative: 'two-sided', 'larger', 'smaller'  
        returns:
            dictionary with z-statistic, p-value, proportions
        """
        from statsmodels.stats.proportion import proportions_ztest
        
        counts = np.array([count1, count2])
        nobs = np.array([nobs1, nobs2])
        
        z_stat, p_value = proportions_ztest(counts, nobs, alternative=alternative)
        
        prop1 = count1 / nobs1 if nobs1 > 0 else 0
        prop2 = count2 / nobs2 if nobs2 > 0 else 0
        
        return {
            'z_statistic': float(z_stat),
            'p_value': float(p_value),
            'proportion_1': prop1,
            'proportion_2': prop2,
            'difference': prop1 - prop2,
            'significant': p_value < self.alpha
        }
    
    def compare_models_scs(
        self, 
        model_results: Dict[str, np.ndarray]
    ) -> Dict[str, any]:
        """
        compare SCS across multiple models using pairwise tests
        
        args:
            model_results: dict mapping model_name -> binary_flip_array   
        returns:
            comprehensive comparison results
        """
        model_names = list(model_results.keys())
        results = {
            'model_means': {},
            'pairwise_comparisons': {},
            'confidence_intervals': {}
        }
        
        #compute means and CIs
        for name, data in model_results.items():
            mean, ci_lower, ci_upper = self.bootstrap_confidence_interval(data)
            results['model_means'][name] = mean
            results['confidence_intervals'][name] = {
                'lower': ci_lower,
                'upper': ci_upper
            }
        
        #pairwise comparisons
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                data1 = model_results[name1]
                data2 = model_results[name2]
        
        return results
    
    def compare_iar_by_family(
        self,
        generative_iar: Tuple[int, int],  #(count_true, total)
        contrastive_iar: Tuple[int, int]
    ) -> Dict[str, any]:
        """
        compare IAR between generative and contrastive model families
        
        args:
            generative_iar: (num_incorrect_agreements, total_tests) for generative
            contrastive_iar: (num_incorrect_agreements, total_tests) for contrastive  
        returns:
            test results
        """
        gen_count, gen_total = generative_iar
        cont_count, cont_total = contrastive_iar
        
        #proportion test
        prop_test = self.proportion_test(
            gen_count, gen_total,
            cont_count, cont_total,
            alternative='larger'  #generative > contrastive
        )
        
        #chi-square test
        contingency = np.array([
            [gen_count, gen_total - gen_count],
            [cont_count, cont_total - cont_count]
        ])
        chi_test = self.chi_square_independence(contingency)
        
        #effect size 
        p1 = gen_count / gen_total
        p2 = cont_count / cont_total
        cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
        
        return {
            'proportion_test': prop_test,
            'chi_square_test': chi_test,
            'cohens_h': cohens_h,
            'effect_size_interpretation': self._interpret_cohens_h(cohens_h)
        }

    #patching tests    
    def permutation_test(
        self,
        experimental_effect: np.ndarray,
        control_effect: np.ndarray,
        n_permutations: int = 10000,
        statistic: str = 'mean_diff'
    ) -> Dict[str, float]:
        """
        permutation test to compare experimental vs control patching effects
        
        args:
            experimental_effect: array of effect sizes from real patches
            control_effect: array of effect sizes from control patches
            n_permutations: number of permutation samples
            statistic: 'mean_diff' or 'median_diff'    
        returns:
            test results including p-value
        """
        stat_fn = np.mean if statistic == 'mean_diff' else np.median
        
        observed_diff = stat_fn(experimental_effect) - stat_fn(control_effect)
        combined = np.concatenate([experimental_effect, control_effect])
        n_exp = len(experimental_effect)
        
        perm_diffs = []
        for _ in range(n_permutations):
            shuffled = np.random.permutation(combined)
            perm_exp = shuffled[:n_exp]
            perm_ctrl = shuffled[n_exp:]
            perm_diffs.append(stat_fn(perm_exp) - stat_fn(perm_ctrl))
        
        perm_diffs = np.array(perm_diffs)
        
        #two-tailed p-value
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        
        return {
            'observed_difference': float(observed_diff),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'permutation_distribution_mean': float(np.mean(perm_diffs)),
            'permutation_distribution_std': float(np.std(perm_diffs))
        }
    
    def patch_success_rate_test(
        self,
        successful_patches: int,
        total_patches: int,
        null_hypothesis_rate: float = 0.5
    ) -> Dict[str, float]:
        """
        test if patch success rate differs from chance (binomial test)
        
        args:
            successful_patches: Number of successful patches
            total_patches: Total patch attempts
            null_hypothesis_rate: Expected rate under null (default: 0.5 = chance)     
        returns:
            binomial test results
        """
        from scipy.stats import binomtest
        result = binomtest(successful_patches, total_patches, null_hypothesis_rate, 
                   alternative='two-sided')
        p_value = result.pvalue
        
        observed_rate = successful_patches / total_patches
        ci_lower, ci_upper = self.wilson_score_interval(successful_patches, total_patches)
        
        return {
            'observed_rate': observed_rate,
            'null_rate': null_hypothesis_rate,
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'confidence_interval': (ci_lower, ci_upper),
            'interpretation': 'above_chance' if observed_rate > null_hypothesis_rate and p_value < self.alpha else 'not_significant'
        }
    
    def compare_patch_success_across_layers(
        self,
        layer_success_rates: Dict[int, Tuple[int, int]] 
    ) -> Dict[str, any]:
        """
        compare patch success rates across different layers
        
        args:
            layer_success_rates: dict mapping layer_id -> (successes, attempts)  
        returns:
            chi-square test and pairwise comparisons
        """
        layers = sorted(layer_success_rates.keys())
        
        #build contingency table
        successes = []
        failures = []
        for layer in layers:
            succ, total = layer_success_rates[layer]
            successes.append(succ)
            failures.append(total - succ)
        
        contingency = np.array([successes, failures])
        chi_test = self.chi_square_independence(contingency) #overall chi-square test
        
        #compute rates
        rates = {
            layer: layer_success_rates[layer][0] / layer_success_rates[layer][1]
            for layer in layers
        }
        
        return {
            'overall_chi_square': chi_test,
            'layer_success_rates': rates,
            'best_layer': max(rates, key=rates.get),
            'best_layer_rate': max(rates.values())
        }
    
    def cosine_shift_significance(
        self,
        experimental_delta_cos: np.ndarray,
        control_delta_cos: np.ndarray
    ) -> Dict[str, float]:
        """
        test if experimental cosine shifts differ from control (for CLIP)
        
        args:
            experimental_delta_cos: Array of Δcos from matched-donor patches
            control_delta_cos: Array of Δcos from random-donor patches    
        returns:
             effect size
        """
        u_stat, p_value = mannwhitneyu(
            experimental_delta_cos, 
            control_delta_cos,
            alternative='greater'  #experimental > control
        )
        
        #effect size (rank-biserial correlation)
        n1 = len(experimental_delta_cos)
        n2 = len(control_delta_cos)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
        
        return {
            'u_statistic': float(u_stat),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'experimental_mean': float(np.mean(experimental_delta_cos)),
            'control_mean': float(np.mean(control_delta_cos)),
            'mean_difference': float(np.mean(experimental_delta_cos) - np.mean(control_delta_cos)),
            'rank_biserial_correlation': float(rank_biserial),
            'effect_size': self._interpret_rank_biserial(rank_biserial)
        }
    
    #helper methods
    def _interpret_rank_biserial(self, r: float) -> str:
        """interpret rank-biserial correlation"""
        abs_r = abs(r)
        if abs_r < 0.3:
            return 'small'
        elif abs_r < 0.5:
            return 'medium'
        else:
            return 'large'
    
    def generate_report(
        self,
        results: Dict[str, any],
        output_path: str = 'statistical_report.json'
    ):
        """save comprehensive statistical report to JSON"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"statistical report saved to {output_path}")

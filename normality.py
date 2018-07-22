import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import random

random.seed(234)
samplesize = [10, 30, 100]
nboot = 500

def gen_means(distribution, samplesize):
    population = distribution.rvs(size=1000)
    samples = []
    means = []
    for n in samplesize:
        # sample without replacement from the population
        sample = np.random.choice(population, n, replace=False)
        bmeans = []
        for i in list(range(0,nboot)):
            # create bootstrapped samples, with replacement
            bsample = np.random.choice(sample, n, replace=True)
            bmeans.append(bsample.mean())
        samples.append(sample)
        bmeans = np.array(bmeans)
        means.append(bmeans)
    samples = np.array(samples)
    means = np.array(means)
    return population, samples, means

def plot_fig(population, samples, means, samplesize, figname):
    fig, ax = plt.subplots(figsize=(11, 7))
    # plot population
    ax1 = plt.subplot(3, 3, (1,3))
    plt.hist(population, bins=100, label='n='+str(len(population)))
    plt.legend(frameon=False)
    plt.xlim([0, 20])
    plt.ylabel('Counts\n(population)')
    plt.xlabel('Units')
    # plot random sample
    ax2 = plt.subplot(3, 3, 4)
    ax2.locator_params(axis='y', nbins=5)
    plt.hist(samples[0], bins=10, label='n='+str(samplesize[0]))
    plt.legend(frameon=False)
    plt.xlim([0, 20])
    plt.ylim([0, 40])
    plt.ylabel('Counts\n(sample values)')
    plt.xlabel('Units')
    ax3 = plt.subplot(3, 3, 5)
    ax3.locator_params(axis='y', nbins=5)
    plt.hist(samples[1], bins=10, label='n='+str(samplesize[1]))
    plt.legend(frameon=False)
    plt.xlim([0, 20])
    plt.ylim([0, 40])
    plt.xlabel('Units')
    ax4 = plt.subplot(3, 3, 6)
    ax4.locator_params(axis='y', nbins=5)
    plt.hist(samples[2], bins=10, label='n='+str(samplesize[2]))
    plt.legend(frameon=False)
    plt.xlim([0, 20])
    plt.ylim([0, 40])
    plt.xlabel('Units')
    # plot bootstrapped sample means
    ax5 = plt.subplot(3, 3, 7)
    ax5.locator_params(axis='y', nbins=5)
    plt.hist(means[0], bins=100, label='n='+str(samplesize[0]))
    plt.legend(frameon=False)
    plt.xlim([0, 20])
    plt.ylim([0, 40])
    plt.ylabel('Counts\n(sample means)')
    plt.xlabel('Units')
    ax6 = plt.subplot(3, 3, 8)
    ax6.locator_params(axis='y', nbins=5)
    plt.hist(means[1], bins=100, label='n='+str(samplesize[1]))
    plt.legend(frameon=False)
    plt.xlim([0, 20])
    plt.ylim([0, 40])
    plt.xlabel('Units')
    ax7 = plt.subplot(3, 3, 9)
    ax7.locator_params(axis='y', nbins=5)
    plt.hist(means[2], bins=100, label='n='+str(samplesize[2]))
    plt.legend(frameon=False)
    plt.xlim([0, 20])
    plt.ylim([0, 40])
    plt.xlabel('Units')
    fig.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()

# Generate normally distributed population, bootstrap sample means and plot

normal_distribution = scipy.stats.norm(scale=3, loc=10) # scale = SD
population, samples, means = gen_means(normal_distribution, samplesize)
plot_fig(population, samples, means, samplesize, 'norm.png')

# Generate skewed population, bootstrap sample means and plot

gamma_distribution = scipy.stats.gamma(3, scale=2, loc=0)
population, samples, means = gen_means(gamma_distribution, samplesize)
plot_fig(population, samples, means, samplesize, 'gamm1.png')

gamma_distribution = scipy.stats.gamma(1.5, scale=2, loc=0)
population, samples, means = gen_means(gamma_distribution, samplesize)
plot_fig(population, samples, means, samplesize, 'gamm2.png')

# Generate power law distributed population, bootstrap sample means and plot

power_distribution = scipy.stats.powerlaw(0.5, scale=20, loc=0)
population, samples, means = gen_means(power_distribution, samplesize)
plot_fig(population, samples, means, samplesize, 'power.png')


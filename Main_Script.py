#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Main Script
Guide Used - 
https://medium.com/python-data/effient-frontier-in-python-34b0c3043314
https://medium.com/python-data/efficient-frontier-portfolio-optimization-with-python-part-2-2-2fe23413ad94
https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
@author: frank
"""
import datetime as dt
import pandas as pd
import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from read_CSV import get_data

plt.style.use('fivethirtyeight')
np.random.seed(777)

def test_split(df, percent=0.99):
    train = df.iloc[0:int(np.round(prices.shape[0]*percent)), :]
    test = df.iloc[int(np.round(prices.shape[0]*percent)): , :]
    return train, test

# Annual Performance Calculation
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


#Random Portfolios  
def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate, num_Stocks):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(num_Stocks)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, num_Stocks):
    results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate, num_Stocks)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=prices.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=prices.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp,2))
    print("Annualised Volatility:", round(sdp,2))
    print("\n")
    print(max_sharpe_allocation)
    print("-"*80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min,2))
    print("Annualised Volatility:", round(sdp_min,2))
    print("\n")
    print(min_vol_allocation)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
    

#Calculated Optimized Portfolios
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result

def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

# Portfolio Stats
def getPortStats(allocs, prices, sv=1):
    normed = normalize_data(prices)
    allocation = normed * allocs
    position_vals = allocation * sv
    port_val = position_vals.sum(axis=1)
    period_return = port_val.pct_change(1)
    period_return = period_return.iloc[1:,]
    
    cr = (port_val.iloc[-1]/port_val.iloc[0]) - 1
    apr = period_return.mean()
    sdpr = period_return.std()
    
    return cr, apr, sdpr, port_val

def normalize_data(df):
        return df / df.iloc[0, :]

def display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, num_Stocks):
    results, _ = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate, num_Stocks)
    
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=prices.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=prices.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp,2))
    print("Annualised Volatility:", round(sdp,2))
    print("\n")
    print(max_sharpe_allocation)
    print("-"*80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min,2))
    print("Annualised Volatility:", round(sdp_min,2))
    print("\n")
    print(min_vol_allocation)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    target = np.linspace(rp_min, 0.32, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
    
def display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate, num_Stocks):
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=prices.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=prices.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns * 252
    
    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp,2))
    print("Annualised Volatility:", round(sdp,2))
    print("\n")
    print(max_sharpe_allocation)
    print("-"*80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min,2))
    print("Annualised Volatility:", round(sdp_min,2))
    print("\n")
    print(min_vol_allocation)
    print("-"*80)
    print("Individual Stock Returns and Volatility\n")
    for i, txt in enumerate(prices.columns):
        print(txt,":","annuaised return",round(an_rt[i],2),", annualised volatility:",round(an_vol[i],2))
    print("-"*80)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(an_vol,an_rt,marker='o',s=200)

    for i, txt in enumerate(prices.columns):
        ax.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')
    ax.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    ax.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    target = np.linspace(rp_min, 0.34, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    ax.set_title('Portfolio Optimization with Individual Stocks')
    ax.set_xlabel('annualised volatility')
    ax.set_ylabel('annualised returns')
    ax.legend(labelspacing=0.8)
    
def display_ef_with_current_alloc(mean_returns, cov_matrix, risk_free_rate, num_Stocks, curAlloc, portTitle, prices):
    #Getting Cumulative Port Value And Sharpe Ratio
    cr, apr, sdpr, port_val = getPortStats(curAlloc, prices, sv=10000)
    
    sr = neg_sharpe_ratio(curAlloc, mean_returns, cov_matrix, risk_free_rate)
    
    sdp, rp = portfolio_annualised_performance(curAlloc, mean_returns, cov_matrix)
    theCurrentAllocation = pd.DataFrame(curAlloc,index=prices.columns,columns=['allocation'])
    theCurrentAllocation.allocation = [round(i*100,2)for i in theCurrentAllocation.allocation]
    theCurrentAllocation = theCurrentAllocation.T
    
    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=prices.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns * 252
    
    print("-"*80)
    print(portTitle + "\n")
    print("Annualised Return:", round(rp,2))
    print("Annualised Volatility:", round(sdp,2))
    print("Sharpe Ratio:", round(sr,2))
    print("Cumulative Returns:", round(cr,2))
    print("Hypothetical Return Of Initial $10,000:", round(port_val.iloc[-1],2))
    
    print("\n")
    print(theCurrentAllocation)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(an_vol,an_rt,marker='o',s=200)
    
    for i, txt in enumerate(prices.columns):
        ax.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')
        
    ax.scatter(sdp,rp,marker='*',color='r',s=500, label=portTitle)
    ax.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
    
    target = np.linspace(rp_min, 0.34, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    ax.set_title('Portfolio Performance On Test Data')
    ax.set_xlabel('annualised volatility')
    ax.set_ylabel('annualised returns')
    ax.legend(labelspacing=0.8)
    # plt.show()

if __name__=="__main__":
    #Setting Dates
    sd=dt.datetime(2005,1,1)
    ed=dt.datetime(2023,7,17)
    dates = pd.date_range(sd,ed)
    
    #Symbols    
    symbols = ['VTV',
               'VBK',
               'VIOO',
               'MGK',
               'VTI',
               'VFH',
               'VGT',
               'VOO',
               'VHT',
               'VOT',
               'VNQ']
               
    prices = get_data(symbols, dates)
    prices = prices.dropna()
    
    #Train and Test split
    pricesTrain, pricesTest = test_split(prices)
    
    
    #Plotting each stock price
    plt.figure(figsize=(14, 7))
    for c in prices.columns.values:
        plt.plot(prices.index, prices[c], lw=3, alpha=0.8,label=c)
    plt.legend(loc='upper left', fontsize=12)
    plt.ylabel('price in $')
   
    #Plotting each daily returns
    returns = prices.pct_change()
    plt.figure(figsize=(14, 7))
    for c in returns.columns.values:
        plt.plot(returns.index, returns[c], lw=3, alpha=0.8,label=c)
    plt.legend(loc='upper right', fontsize=12)
    plt.ylabel('daily returns')
    
    #Defining Arg values
    #Train Data
    returns = pricesTrain.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_portfolios = 25000
    risk_free_rate = 0.0381
    num_Stocks = len(symbols)
    
    #Current Allocation
    curAlloc = np.array([0.0,
                         0.0,
                         0.0,
                         0.0,
                         1.0,
                         0.0,
                         0.0,
                         0.0,
                         0.0,
                         0.0,
                         0.0])
        
#    #Suggested Alloc
#    Suggested_Alloc = np.array([0.1460,
#                         0.1000,
#                         0.0930,
#                         0.0930,
#                         0.1770,
#                         0.1920,
#                         0.1430,
#                         0.0390,
#                         0.0170])
    
    #Naive Alloc
    Suggested_Alloc = np.empty(len(symbols), float)
    Suggested_Alloc.fill(1.0/len(symbols))
        
    #Getting Max Sharpe Ratio Alloc
    curAlloc_max_sharpe_ratio = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)['x']
    
    #Getting Min Vol Alloc
    curAlloc_min_variance = min_variance(mean_returns, cov_matrix)['x']
    
    
    #Simulated EF with Random
    display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, num_Stocks)
    
    #Calculated Optimized EF
    display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, num_Stocks)
    
    #Plot each individual stocks with corresponding values of each stock's annual return and annual risk
    display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate, num_Stocks)


#    #Compare all on train data
#    display_ef_with_current_alloc(mean_returns, cov_matrix, risk_free_rate, num_Stocks, curAlloc, "Current Alloc", pricesTrain)
#    display_ef_with_current_alloc(mean_returns, cov_matrix, risk_free_rate, num_Stocks, Suggested_Alloc, "Suggested Alloc", pricesTrain)
#    display_ef_with_current_alloc(mean_returns, cov_matrix, risk_free_rate, num_Stocks, curAlloc_max_sharpe_ratio, "Maximum Sharpe Ratio Portfolio Allocation", pricesTrain)
#    display_ef_with_current_alloc(mean_returns, cov_matrix, risk_free_rate, num_Stocks, curAlloc_min_variance, "Minimum Volatility Portfolio Allocation", pricesTrain)


    
    #Compare all of them on test data
    returns = pricesTest.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    display_ef_with_current_alloc(mean_returns, cov_matrix, risk_free_rate, num_Stocks, curAlloc, "Current Alloc", pricesTest)
    display_ef_with_current_alloc(mean_returns, cov_matrix, risk_free_rate, num_Stocks, Suggested_Alloc, "Suggested Alloc", pricesTest)
    display_ef_with_current_alloc(mean_returns, cov_matrix, risk_free_rate, num_Stocks, curAlloc_max_sharpe_ratio, "Maximum Sharpe Ratio Portfolio Allocation", pricesTest)
    display_ef_with_current_alloc(mean_returns, cov_matrix, risk_free_rate, num_Stocks, curAlloc_min_variance, "Minimum Volatility Portfolio Allocation", pricesTest)

/**
 * BRICS DeFi Smart Contract for Hyperledger Fabric
 * 
 * This contract implements the core functionality for the BRICS DeFi ecosystem,
 * including cross-border payments, liquidity pools, and governance.
 */

'use strict';

const { Contract } = require('fabric-contract-api');

class BRICSDefiContract extends Contract {
    /**
     * Initialize the ledger with default values
     * @param {Context} ctx - The transaction context
     */
    async init(ctx) {
        console.info('============= Initializing BRICS DeFi Contract =============');
        
        // Initialize supported currencies
        const currencies = {
            'BRL': { name: 'Brazilian Real', country: 'Brazil', exchangeRate: 5.0 },
            'RUB': { name: 'Russian Ruble', country: 'Russia', exchangeRate: 75.0 },
            'INR': { name: 'Indian Rupee', country: 'India', exchangeRate: 83.0 },
            'CNY': { name: 'Chinese Yuan', country: 'China', exchangeRate: 7.2 },
            'ZAR': { name: 'South African Rand', country: 'South Africa', exchangeRate: 18.5 },
            'BRICS': { name: 'BRICS Token', country: 'All', exchangeRate: 1.0 }
        };
        
        await ctx.stub.putState('currencies', Buffer.from(JSON.stringify(currencies)));
        
        // Initialize governance parameters
        const governance = {
            votingThreshold: 0.66, // 66% majority required
            proposalDuration: 7 * 24 * 60 * 60, // 7 days in seconds
            minimumStake: 1000, // Minimum BRICS tokens to create proposal
            activeProposals: []
        };
        
        await ctx.stub.putState('governance', Buffer.from(JSON.stringify(governance)));
        
        // Initialize liquidity pools
        const liquidityPools = {
            'BRL_BRICS': { token1: 'BRL', token2: 'BRICS', reserve1: 1000000, reserve2: 1000000, fee: 0.003 },
            'RUB_BRICS': { token1: 'RUB', token2: 'BRICS', reserve1: 1000000, reserve2: 1000000, fee: 0.003 },
            'INR_BRICS': { token1: 'INR', token2: 'BRICS', reserve1: 1000000, reserve2: 1000000, fee: 0.003 },
            'CNY_BRICS': { token1: 'CNY', token2: 'BRICS', reserve1: 1000000, reserve2: 1000000, fee: 0.003 },
            'ZAR_BRICS': { token1: 'ZAR', token2: 'BRICS', reserve1: 1000000, reserve2: 1000000, fee: 0.003 }
        };
        
        await ctx.stub.putState('liquidityPools', Buffer.from(JSON.stringify(liquidityPools)));
        
        console.info('============= BRICS DeFi Contract Initialized =============');
    }
    
    /**
     * Create a user account
     * @param {Context} ctx - The transaction context
     * @param {String} userId - User ID
     * @param {String} country - User's country (must be a BRICS member)
     * @param {String} name - User's name
     */
    async createAccount(ctx, userId, country, name) {
        console.info('============= Creating Account =============');
        
        // Check if country is a BRICS member
        const validCountries = ['Brazil', 'Russia', 'India', 'China', 'South Africa'];
        if (!validCountries.includes(country)) {
            throw new Error(`Country ${country} is not a BRICS member`);
        }
        
        // Check if account already exists
        const accountAsBytes = await ctx.stub.getState(userId);
        if (accountAsBytes && accountAsBytes.length > 0) {
            throw new Error(`Account ${userId} already exists`);
        }
        
        // Create account with initial balances
        const account = {
            userId,
            country,
            name,
            balances: {
                'BRICS': 100, // Initial BRICS tokens
                'BRL': 0,
                'RUB': 0,
                'INR': 0,
                'CNY': 0,
                'ZAR': 0
            },
            transactions: [],
            createdAt: ctx.stub.getTxTimestamp().seconds
        };
        
        await ctx.stub.putState(userId, Buffer.from(JSON.stringify(account)));
        
        return JSON.stringify(account);
    }
    
    /**
     * Get account information
     * @param {Context} ctx - The transaction context
     * @param {String} userId - User ID
     */
    async getAccount(ctx, userId) {
        const accountAsBytes = await ctx.stub.getState(userId);
        if (!accountAsBytes || accountAsBytes.length === 0) {
            throw new Error(`Account ${userId} does not exist`);
        }
        
        return accountAsBytes.toString();
    }
    
    /**
     * Transfer tokens between accounts
     * @param {Context} ctx - The transaction context
     * @param {String} senderId - Sender's user ID
     * @param {String} receiverId - Receiver's user ID
     * @param {String} currency - Currency to transfer
     * @param {Number} amount - Amount to transfer
     */
    async transfer(ctx, senderId, receiverId, currency, amount) {
        amount = parseFloat(amount);
        if (isNaN(amount) || amount <= 0) {
            throw new Error('Amount must be a positive number');
        }
        
        // Get sender account
        const senderAccountAsBytes = await ctx.stub.getState(senderId);
        if (!senderAccountAsBytes || senderAccountAsBytes.length === 0) {
            throw new Error(`Sender account ${senderId} does not exist`);
        }
        const senderAccount = JSON.parse(senderAccountAsBytes.toString());
        
        // Get receiver account
        const receiverAccountAsBytes = await ctx.stub.getState(receiverId);
        if (!receiverAccountAsBytes || receiverAccountAsBytes.length === 0) {
            throw new Error(`Receiver account ${receiverId} does not exist`);
        }
        const receiverAccount = JSON.parse(receiverAccountAsBytes.toString());
        
        // Check if currency exists
        const currenciesAsBytes = await ctx.stub.getState('currencies');
        const currencies = JSON.parse(currenciesAsBytes.toString());
        if (!currencies[currency]) {
            throw new Error(`Currency ${currency} is not supported`);
        }
        
        // Check if sender has sufficient balance
        if (!senderAccount.balances[currency] || senderAccount.balances[currency] < amount) {
            throw new Error(`Insufficient ${currency} balance`);
        }
        
        // Update balances
        senderAccount.balances[currency] -= amount;
        if (!receiverAccount.balances[currency]) {
            receiverAccount.balances[currency] = 0;
        }
        receiverAccount.balances[currency] += amount;
        
        // Record transaction in both accounts
        const txId = ctx.stub.getTxID();
        const timestamp = ctx.stub.getTxTimestamp().seconds;
        
        const transaction = {
            txId,
            type: 'transfer',
            currency,
            amount,
            counterparty: receiverId,
            timestamp
        };
        
        senderAccount.transactions.push(transaction);
        
        const receiverTransaction = {
            txId,
            type: 'receive',
            currency,
            amount,
            counterparty: senderId,
            timestamp
        };
        
        receiverAccount.transactions.push(receiverTransaction);
        
        // Update state
        await ctx.stub.putState(senderId, Buffer.from(JSON.stringify(senderAccount)));
        await ctx.stub.putState(receiverId, Buffer.from(JSON.stringify(receiverAccount)));
        
        return JSON.stringify(transaction);
    }
    
    /**
     * Swap tokens using liquidity pools
     * @param {Context} ctx - The transaction context
     * @param {String} userId - User ID
     * @param {String} fromCurrency - Currency to swap from
     * @param {String} toCurrency - Currency to swap to
     * @param {Number} amount - Amount to swap
     */
    async swap(ctx, userId, fromCurrency, toCurrency, amount) {
        amount = parseFloat(amount);
        if (isNaN(amount) || amount <= 0) {
            throw new Error('Amount must be a positive number');
        }
        
        // Get user account
        const accountAsBytes = await ctx.stub.getState(userId);
        if (!accountAsBytes || accountAsBytes.length === 0) {
            throw new Error(`Account ${userId} does not exist`);
        }
        const account = JSON.parse(accountAsBytes.toString());
        
        // Check if user has sufficient balance
        if (!account.balances[fromCurrency] || account.balances[fromCurrency] < amount) {
            throw new Error(`Insufficient ${fromCurrency} balance`);
        }
        
        // Get liquidity pools
        const liquidityPoolsAsBytes = await ctx.stub.getState('liquidityPools');
        const liquidityPools = JSON.parse(liquidityPoolsAsBytes.toString());
        
        // Find the appropriate pool
        const poolKey = `${fromCurrency}_${toCurrency}`;
        const reversePoolKey = `${toCurrency}_${fromCurrency}`;
        
        let pool;
        let isReverse = false;
        
        if (liquidityPools[poolKey]) {
            pool = liquidityPools[poolKey];
        } else if (liquidityPools[reversePoolKey]) {
            pool = liquidityPools[reversePoolKey];
            isReverse = true;
        } else {
            throw new Error(`No liquidity pool exists for ${fromCurrency} to ${toCurrency}`);
        }
        
        // Calculate swap amount using constant product formula (x * y = k)
        let inputReserve, outputReserve;
        
        if (!isReverse) {
            inputReserve = pool.reserve1;
            outputReserve = pool.reserve2;
        } else {
            inputReserve = pool.reserve2;
            outputReserve = pool.reserve1;
        }
        
        // Calculate output amount: (outputReserve * inputAmount) / (inputReserve + inputAmount)
        const inputAmountWithFee = amount * (1 - pool.fee);
        const outputAmount = (outputReserve * inputAmountWithFee) / (inputReserve + inputAmountWithFee);
        
        // Update pool reserves
        if (!isReverse) {
            pool.reserve1 += amount;
            pool.reserve2 -= outputAmount;
        } else {
            pool.reserve2 += amount;
            pool.reserve1 -= outputAmount;
        }
        
        // Update user balances
        account.balances[fromCurrency] -= amount;
        if (!account.balances[toCurrency]) {
            account.balances[toCurrency] = 0;
        }
        account.balances[toCurrency] += outputAmount;
        
        // Record transaction
        const txId = ctx.stub.getTxID();
        const timestamp = ctx.stub.getTxTimestamp().seconds;
        
        const transaction = {
            txId,
            type: 'swap',
            fromCurrency,
            toCurrency,
            inputAmount: amount,
            outputAmount,
            timestamp
        };
        
        account.transactions.push(transaction);
        
        // Update state
        await ctx.stub.putState(userId, Buffer.from(JSON.stringify(account)));
        await ctx.stub.putState('liquidityPools', Buffer.from(JSON.stringify(liquidityPools)));
        
        return JSON.stringify({
            fromCurrency,
            toCurrency,
            inputAmount: amount,
            outputAmount
        });
    }
    
    /**
     * Add liquidity to a pool
     * @param {Context} ctx - The transaction context
     * @param {String} userId - User ID
     * @param {String} currency1 - First currency
     * @param {String} currency2 - Second currency
     * @param {Number} amount1 - Amount of first currency
     * @param {Number} amount2 - Amount of second currency
     */
    async addLiquidity(ctx, userId, currency1, currency2, amount1, amount2) {
        amount1 = parseFloat(amount1);
        amount2 = parseFloat(amount2);
        
        if (isNaN(amount1) || amount1 <= 0 || isNaN(amount2) || amount2 <= 0) {
            throw new Error('Amounts must be positive numbers');
        }
        
        // Get user account
        const accountAsBytes = await ctx.stub.getState(userId);
        if (!accountAsBytes || accountAsBytes.length === 0) {
            throw new Error(`Account ${userId} does not exist`);
        }
        const account = JSON.parse(accountAsBytes.toString());
        
        // Check balances
        if (!account.balances[currency1] || account.balances[currency1] < amount1) {
            throw new Error(`Insufficient ${currency1} balance`);
        }
        
        if (!account.balances[currency2] || account.balances[currency2] < amount2) {
            throw new Error(`Insufficient ${currency2} balance`);
        }
        
        // Get liquidity pools
        const liquidityPoolsAsBytes = await ctx.stub.getState('liquidityPools');
        const liquidityPools = JSON.parse(liquidityPoolsAsBytes.toString());
        
        // Find or create pool
        const poolKey = `${currency1}_${currency2}`;
        const reversePoolKey = `${currency2}_${currency1}`;
        
        let pool;
        let isReverse = false;
        
        if (liquidityPools[poolKey]) {
            pool = liquidityPools[poolKey];
        } else if (liquidityPools[reversePoolKey]) {
            pool = liquidityPools[reversePoolKey];
            isReverse = true;
        } else {
            // Create new pool
            pool = {
                token1: currency1,
                token2: currency2,
                reserve1: 0,
                reserve2: 0,
                fee: 0.003 // 0.3% fee
            };
            liquidityPools[poolKey] = pool;
        }
        
        // Update pool reserves
        if (!isReverse) {
            pool.reserve1 += amount1;
            pool.reserve2 += amount2;
        } else {
            pool.reserve1 += amount2;
            pool.reserve2 += amount1;
        }
        
        // Update user balances
        account.balances[currency1] -= amount1;
        account.balances[currency2] -= amount2;
        
        // Record transaction
        const txId = ctx.stub.getTxID();
        const timestamp = ctx.stub.getTxTimestamp().seconds;
        
        const transaction = {
            txId,
            type: 'addLiquidity',
            currency1,
            currency2,
            amount1,
            amount2,
            timestamp
        };
        
        account.transactions.push(transaction);
        
        // Update state
        await ctx.stub.putState(userId, Buffer.from(JSON.stringify(account)));
        await ctx.stub.putState('liquidityPools', Buffer.from(JSON.stringify(liquidityPools)));
        
        return JSON.stringify(transaction);
    }
    
    /**
     * Create a governance proposal
     * @param {Context} ctx - The transaction context
     * @param {String} userId - User ID
     * @param {String} title - Proposal title
     * @param {String} description - Proposal description
     * @param {String} proposalType - Type of proposal
     * @param {String} parameters - JSON string of proposal parameters
     */
    async createProposal(ctx, userId, title, description, proposalType, parameters) {
        // Get user account
        const accountAsBytes = await ctx.stub.getState(userId);
        if (!accountAsBytes || accountAsBytes.length === 0) {
            throw new Error(`Account ${userId} does not exist`);
        }
        const account = JSON.parse(accountAsBytes.toString());
        
        // Get governance parameters
        const governanceAsBytes = await ctx.stub.getState('governance');
        const governance = JSON.parse(governanceAsBytes.toString());
        
        // Check if user has enough stake
        if (!account.balances.BRICS || account.balances.BRICS < governance.minimumStake) {
            throw new Error(`Insufficient BRICS tokens for proposal creation (minimum: ${governance.minimumStake})`);
        }
        
        // Create proposal
        const proposalId = ctx.stub.getTxID();
        const timestamp = ctx.stub.getTxTimestamp().seconds;
        const endTime = timestamp + governance.proposalDuration;
        
        const proposal = {
            proposalId,
            title,
            description,
            proposalType,
            parameters: JSON.parse(parameters),
            creator: userId,
            createdAt: timestamp,
            endTime,
            votes: {
                yes: 0,
                no: 0
            },
            voters: {},
            status: 'active'
        };
        
        // Add to active proposals
        governance.activeProposals.push(proposalId);
        
        // Store proposal
        await ctx.stub.putState(proposalId, Buffer.from(JSON.stringify(proposal)));
        await ctx.stub.putState('governance', Buffer.from(JSON.stringify(governance)));
        
        return JSON.stringify(proposal);
    }
    
    /**
     * Vote on a governance proposal
     * @param {Context} ctx - The transaction context
     * @param {String} userId - User ID
     * @param {String} proposalId - Proposal ID
     * @param {String} vote - 'yes' or 'no'
     */
    async voteOnProposal(ctx, userId, proposalId, vote) {
        if (vote !== 'yes' && vote !== 'no') {
            throw new Error("Vote must be 'yes' or 'no'");
        }
        
        // Get user account
        const accountAsBytes = await ctx.stub.getState(userId);
        if (!accountAsBytes || accountAsBytes.length === 0) {
            throw new Error(`Account ${userId} does not exist`);
        }
        const account = JSON.parse(accountAsBytes.toString());
        
        // Get proposal
        const proposalAsBytes = await ctx.stub.getState(proposalId);
        if (!proposalAsBytes || proposalAsBytes.length === 0) {
            throw new Error(`Proposal ${proposalId} does not exist`);
        }
        const proposal = JSON.parse(proposalAsBytes.toString());
        
        // Check if proposal is active
        if (proposal.status !== 'active') {
            throw new Error(`Proposal ${proposalId} is not active`);
        }
        
        // Check if voting period has ended
        const timestamp = ctx.stub.getTxTimestamp().seconds;
        if (timestamp > proposal.endTime) {
            throw new Error(`Voting period for proposal ${proposalId} has ended`);
        }
        
        // Check if user has already voted
        if (proposal.voters[userId]) {
            throw new Error(`User ${userId} has already voted on this proposal`);
        }
        
        // Calculate voting power (based on BRICS token balance)
        const votingPower = account.balances.BRICS || 0;
        if (votingPower <= 0) {
            throw new Error(`User ${userId} has no voting power (BRICS tokens)`);
        }
        
        // Record vote
        proposal.votes[vote] += votingPower;
        proposal.voters[userId] = {
            vote,
            power: votingPower,
            timestamp
        };
        
        // Update proposal
        await ctx.stub.putState(proposalId, Buffer.from(JSON.stringify(proposal)));
        
        return JSON.stringify({
            proposalId,
            vote,
            votingPower
        });
    }
    
    /**
     * Execute a proposal if it has passed
     * @param {Context} ctx - The transaction context
     * @param {String} proposalId - Proposal ID
     */
    async executeProposal(ctx, proposalId) {
        // Get proposal
        const proposalAsBytes = await ctx.stub.getState(proposalId);
        if (!proposalAsBytes || proposalAsBytes.length === 0) {
            throw new Error(`Proposal ${proposalId} does not exist`);
        }
        const proposal = JSON.parse(proposalAsBytes.toString());
        
        // Check if proposal is active
        if (proposal.status !== 'active') {
            throw new Error(`Proposal ${proposalId} is not active`);
        }
        
        // Check if voting period has ended
        const timestamp = ctx.stub.getTxTimestamp().seconds;
        if (timestamp <= proposal.endTime) {
            throw new Error(`Voting period for proposal ${proposalId} has not ended yet`);
        }
        
        // Get governance parameters
        const governanceAsBytes = await ctx.stub.getState('governance');
        const governance = JSON.parse(governanceAsBytes.toString());
        
        // Calculate total votes
        const totalVotes = proposal.votes.yes + proposal.votes.no;
        if (totalVotes === 0) {
            proposal.status = 'rejected';
            await ctx.stub.putState(proposalId, Buffer.from(JSON.stringify(proposal)));
            throw new Error(`Proposal ${proposalId} has no votes`);
        }
        
        // Check if proposal passed
        const yesPercentage = proposal.votes.yes / totalVotes;
        if (yesPercentage >= governance.votingThreshold) {
            // Execute proposal based on type
            if (proposal.proposalType === 'updateFee') {
                await this._executeUpdateFeeProposal(ctx, proposal);
            } else if (proposal.proposalType === 'addCurrency') {
                await this._executeAddCurrencyProposal(ctx, proposal);
            } else if (proposal.proposalType === 'updateGovernance') {
                await this._executeUpdateGovernanceProposal(ctx, proposal);
            } else {
                throw new Error(`Unknown proposal type: ${proposal.proposalType}`);
            }
            
            proposal.status = 'executed';
        } else {
            proposal.status = 'rejected';
        }
        
        // Remove from active proposals
        governance.activeProposals = governance.activeProposals.filter(id => id !== proposalId);
        
        // Update state
        await ctx.stub.putState(proposalId, Buffer.from(JSON.stringify(proposal)));
        await ctx.stub.putState('governance', Buffer.from(JSON.stringify(governance)));
        
        return JSON.stringify({
            proposalId,
            status: proposal.status,
            yesPercentage,
            threshold: governance.votingThreshold
        });
    }
    
    /**
     * Execute a proposal to update pool fees
     * @param {Context} ctx - The transaction context
     * @param {Object} proposal - The proposal object
     */
    async _executeUpdateFeeProposal(ctx, proposal) {
        const { poolKey, newFee } = proposal.parameters;
        
        // Get liquidity pools
        const liquidityPoolsAsBytes = await ctx.stub.getState('liquidityPools');
        const liquidityPools = JSON.parse(liquidityPoolsAsBytes.toString());
        
        // Check if pool exists
        if (!liquidityPools[poolKey]) {
            throw new Error(`Liquidity pool ${poolKey} does not exist`);
        }
        
        // Update fee
        liquidityPools[poolKey].fee = newFee;
        
        // Update state
        await ctx.stub.putState('liquidityPools', Buffer.from(JSON.stringify(liquidityPools)));
    }
    
    /**
     * Execute a proposal to add a new currency
     * @param {Context} ctx - The transaction context
     * @param {Object} proposal - The proposal object
     */
    async _executeAddCurrencyProposal(ctx, proposal) {
        const { currencyCode, name, country, exchangeRate } = proposal.parameters;
        
        // Get currencies
        const currenciesAsBytes = await ctx.stub.getState('currencies');
        const currencies = JSON.parse(currenciesAsBytes.toString());
        
        // Check if currency already exists
        if (currencies[currencyCode]) {
            throw new Error(`Currency ${currencyCode} already exists`);
        }
        
        // Add currency
        currencies[currencyCode] = {
            name,
            country,
            exchangeRate
        };
        
        // Update state
        await ctx.stub.putState('currencies', Buffer.from(JSON.stringify(currencies)));
    }
    
    /**
     * Execute a proposal to update governance parameters
     * @param {Context} ctx - The transaction context
     * @param {Object} proposal - The proposal object
     */
    async _executeUpdateGovernanceProposal(ctx, proposal) {
        const { votingThreshold, proposalDuration, minimumStake } = proposal.parameters;
        
        // Get governance parameters
        const governanceAsBytes = await ctx.stub.getState('governance');
        const governance = JSON.parse(governanceAsBytes.toString());
        
        // Update parameters
        if (votingThreshold !== undefined) governance.votingThreshold = votingThreshold;
        if (proposalDuration !== undefined) governance.proposalDuration = proposalDuration;
        if (minimumStake !== undefined) governance.minimumStake = minimumStake;
        
        // Update state
        await ctx.stub.putState('governance', Buffer.from(JSON.stringify(governance)));
    }
}

module.exports = BRICSDefiContract; 
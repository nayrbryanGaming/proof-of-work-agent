use anchor_lang::prelude::*;
use anchor_lang::solana_program::clock::Clock;

declare_id!("4Jg9fBRCYguygWsYzZEuFaDfpWH5Ffxa7LQLwnatuh5r");

/// POW Bounty Program - Production-Grade Anchor Program
/// 
/// Features:
/// - Create bounties with rewards
/// - Submit proof-of-work hashes
/// - Track submission history
/// - Verify and pay solvers
/// - Event emission for indexing
/// 
/// PDA Seeds: ["bounty", bounty_id.to_le_bytes()]

#[program]
pub mod pow_bounty {
    use super::*;

    /// Initialize the program state (admin only, once)
    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        let state = &mut ctx.accounts.program_state;
        state.admin = ctx.accounts.admin.key();
        state.total_bounties = 0;
        state.total_submissions = 0;
        state.total_paid = 0;
        state.initialized_at = Clock::get()?.unix_timestamp;
        
        emit!(ProgramInitialized {
            admin: state.admin,
            timestamp: state.initialized_at,
        });
        
        Ok(())
    }

    /// Create a new bounty
    pub fn create_bounty(
        ctx: Context<CreateBounty>,
        id: u64,
        description: String,
        reward: u64,
        deadline: Option<i64>,
    ) -> Result<()> {
        require!(description.len() <= Bounty::MAX_DESCRIPTION, PowError::DescriptionTooLong);
        require!(id > 0, PowError::InvalidBountyId);
        
        let bounty = &mut ctx.accounts.bounty;
        let clock = Clock::get()?;
        
        bounty.id = id;
        bounty.creator = ctx.accounts.authority.key();
        bounty.description = description.clone();
        bounty.reward = reward;
        bounty.deadline = deadline;
        bounty.status = BountyStatus::Open;
        bounty.solver = None;
        bounty.result_hash = None;
        bounty.created_at = clock.unix_timestamp;
        bounty.solved_at = None;
        bounty.submission_count = 0;
        bounty.bump = ctx.bumps.bounty;
        
        // Update program state
        let state = &mut ctx.accounts.program_state;
        state.total_bounties = state.total_bounties.checked_add(1).unwrap();
        
        emit!(BountyCreated {
            bounty_id: id,
            creator: ctx.accounts.authority.key(),
            description,
            reward,
            deadline,
            timestamp: clock.unix_timestamp,
        });
        
        msg!("Bounty {} created with reward {} lamports", id, reward);
        Ok(())
    }

    /// Submit proof-of-work for a bounty
    pub fn submit_work(
        ctx: Context<SubmitWork>,
        result_hash: String,
    ) -> Result<()> {
        require!(result_hash.len() == 64, PowError::InvalidHashLength);
        
        let bounty = &mut ctx.accounts.bounty;
        let clock = Clock::get()?;
        
        // Check bounty is still open
        require!(bounty.status == BountyStatus::Open, PowError::BountyNotOpen);
        
        // Check deadline if set
        if let Some(deadline) = bounty.deadline {
            require!(clock.unix_timestamp <= deadline, PowError::DeadlinePassed);
        }
        
        // Record submission
        bounty.status = BountyStatus::Submitted;
        bounty.solver = Some(ctx.accounts.solver.key());
        bounty.result_hash = Some(result_hash.clone());
        bounty.solved_at = Some(clock.unix_timestamp);
        bounty.submission_count = bounty.submission_count.checked_add(1).unwrap();
        
        // Update program state
        let state = &mut ctx.accounts.program_state;
        state.total_submissions = state.total_submissions.checked_add(1).unwrap();
        
        emit!(WorkSubmitted {
            bounty_id: bounty.id,
            solver: ctx.accounts.solver.key(),
            result_hash,
            timestamp: clock.unix_timestamp,
        });
        
        msg!("Work submitted for bounty {}", bounty.id);
        Ok(())
    }

    /// Verify submission and pay the solver
    pub fn verify_and_pay(ctx: Context<VerifyAndPay>) -> Result<()> {
        let bounty = &mut ctx.accounts.bounty;
        
        // Verify bounty state
        require!(bounty.status == BountyStatus::Submitted, PowError::NotSubmitted);
        require!(bounty.creator == ctx.accounts.authority.key(), PowError::NotCreator);
        
        // Verify solver matches
        let expected_solver = bounty.solver.ok_or(PowError::NoSolver)?;
        require!(expected_solver == ctx.accounts.solver.key(), PowError::SolverMismatch);
        
        // Transfer reward if any
        if bounty.reward > 0 {
            let ix = anchor_lang::solana_program::system_instruction::transfer(
                ctx.accounts.authority.key,
                ctx.accounts.solver.key,
                bounty.reward,
            );
            anchor_lang::solana_program::program::invoke(
                &ix,
                &[
                    ctx.accounts.authority.to_account_info(),
                    ctx.accounts.solver.to_account_info(),
                    ctx.accounts.system_program.to_account_info(),
                ],
            )?;
        }
        
        // Update bounty status
        bounty.status = BountyStatus::Completed;
        
        // Update program state
        let state = &mut ctx.accounts.program_state;
        state.total_paid = state.total_paid.checked_add(bounty.reward).unwrap();
        
        emit!(BountyCompleted {
            bounty_id: bounty.id,
            solver: ctx.accounts.solver.key(),
            reward: bounty.reward,
            timestamp: Clock::get()?.unix_timestamp,
        });
        
        msg!("Bounty {} completed, paid {} lamports", bounty.id, bounty.reward);
        Ok(())
    }

    /// Cancel a bounty (creator only, if not yet solved)
    pub fn cancel_bounty(ctx: Context<CancelBounty>) -> Result<()> {
        let bounty = &mut ctx.accounts.bounty;
        
        require!(bounty.status == BountyStatus::Open, PowError::BountyNotOpen);
        require!(bounty.creator == ctx.accounts.authority.key(), PowError::NotCreator);
        
        bounty.status = BountyStatus::Cancelled;
        
        emit!(BountyCancelled {
            bounty_id: bounty.id,
            timestamp: Clock::get()?.unix_timestamp,
        });
        
        msg!("Bounty {} cancelled", bounty.id);
        Ok(())
    }
}

// ===========================================================
// Account Contexts
// ===========================================================

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = admin,
        seeds = [b"program_state"],
        bump,
        space = 8 + ProgramState::SPACE
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub admin: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(id: u64)]
pub struct CreateBounty<'info> {
    #[account(
        init,
        payer = authority,
        seeds = [b"bounty", &id.to_le_bytes()],
        bump,
        space = 8 + Bounty::SPACE
    )]
    pub bounty: Account<'info, Bounty>,
    
    #[account(
        mut,
        seeds = [b"program_state"],
        bump
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct SubmitWork<'info> {
    #[account(
        mut,
        seeds = [b"bounty", &bounty.id.to_le_bytes()],
        bump = bounty.bump
    )]
    pub bounty: Account<'info, Bounty>,
    
    #[account(
        mut,
        seeds = [b"program_state"],
        bump
    )]
    pub program_state: Account<'info, ProgramState>,
    
    pub solver: Signer<'info>,
}

#[derive(Accounts)]
pub struct VerifyAndPay<'info> {
    #[account(
        mut,
        seeds = [b"bounty", &bounty.id.to_le_bytes()],
        bump = bounty.bump
    )]
    pub bounty: Account<'info, Bounty>,
    
    #[account(
        mut,
        seeds = [b"program_state"],
        bump
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    /// CHECK: Validated against bounty.solver
    #[account(mut)]
    pub solver: SystemAccount<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct CancelBounty<'info> {
    #[account(
        mut,
        seeds = [b"bounty", &bounty.id.to_le_bytes()],
        bump = bounty.bump
    )]
    pub bounty: Account<'info, Bounty>,
    
    pub authority: Signer<'info>,
}

// ===========================================================
// Account Data Structures
// ===========================================================

#[account]
pub struct ProgramState {
    pub admin: Pubkey,
    pub total_bounties: u64,
    pub total_submissions: u64,
    pub total_paid: u64,
    pub initialized_at: i64,
}

impl ProgramState {
    pub const SPACE: usize = 32 + 8 + 8 + 8 + 8;
}

#[account]
pub struct Bounty {
    pub id: u64,
    pub creator: Pubkey,
    pub description: String,
    pub reward: u64,
    pub deadline: Option<i64>,
    pub status: BountyStatus,
    pub solver: Option<Pubkey>,
    pub result_hash: Option<String>,
    pub created_at: i64,
    pub solved_at: Option<i64>,
    pub submission_count: u64,
    pub bump: u8,
}

impl Bounty {
    pub const MAX_DESCRIPTION: usize = 280;
    pub const MAX_HASH: usize = 64;
    
    pub const SPACE: usize = 
        8 +                              // id
        32 +                             // creator
        4 + Self::MAX_DESCRIPTION +      // description
        8 +                              // reward
        1 + 8 +                          // deadline (Option<i64>)
        1 +                              // status
        1 + 32 +                         // solver (Option<Pubkey>)
        1 + 4 + Self::MAX_HASH +         // result_hash (Option<String>)
        8 +                              // created_at
        1 + 8 +                          // solved_at (Option<i64>)
        8 +                              // submission_count
        1;                               // bump
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq)]
pub enum BountyStatus {
    Open,
    Submitted,
    Completed,
    Cancelled,
}

impl Default for BountyStatus {
    fn default() -> Self {
        BountyStatus::Open
    }
}

// ===========================================================
// Events
// ===========================================================

#[event]
pub struct ProgramInitialized {
    pub admin: Pubkey,
    pub timestamp: i64,
}

#[event]
pub struct BountyCreated {
    pub bounty_id: u64,
    pub creator: Pubkey,
    pub description: String,
    pub reward: u64,
    pub deadline: Option<i64>,
    pub timestamp: i64,
}

#[event]
pub struct WorkSubmitted {
    pub bounty_id: u64,
    pub solver: Pubkey,
    pub result_hash: String,
    pub timestamp: i64,
}

#[event]
pub struct BountyCompleted {
    pub bounty_id: u64,
    pub solver: Pubkey,
    pub reward: u64,
    pub timestamp: i64,
}

#[event]
pub struct BountyCancelled {
    pub bounty_id: u64,
    pub timestamp: i64,
}

// ===========================================================
// Error Codes
// ===========================================================

#[error_code]
pub enum PowError {
    #[msg("Bounty not solved yet")]
    NotSolved,
    
    #[msg("Description too long (max 280 chars)")]
    DescriptionTooLong,
    
    #[msg("Invalid bounty ID")]
    InvalidBountyId,
    
    #[msg("Invalid hash length (must be 64 hex chars)")]
    InvalidHashLength,
    
    #[msg("Bounty is not open")]
    BountyNotOpen,
    
    #[msg("Deadline has passed")]
    DeadlinePassed,
    
    #[msg("Work not submitted")]
    NotSubmitted,
    
    #[msg("Only creator can perform this action")]
    NotCreator,
    
    #[msg("No solver assigned")]
    NoSolver,
    
    #[msg("Solver address mismatch")]
    SolverMismatch,
}

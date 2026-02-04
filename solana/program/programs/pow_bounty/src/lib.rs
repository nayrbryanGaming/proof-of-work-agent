use anchor_lang::prelude::*;

declare_id!("4Jg9fBRCYguygWsYzZEuFaDfpWH5Ffxa7LQLwnatuh5r");

#[program]
pub mod pow_bounty {
    use super::*;

    pub fn create_bounty(
        ctx: Context<CreateBounty>,
        id: u64,
        description: String,
        reward: u64,
    ) -> Result<()> {
        let bounty = &mut ctx.accounts.bounty;
        bounty.id = id;
        bounty.description = description;
        bounty.reward = reward;
        bounty.solved = false;
        Ok(())
    }

    pub fn submit_work(ctx: Context<SubmitWork>, result_hash: String) -> Result<()> {
        let bounty = &mut ctx.accounts.bounty;
        bounty.solved = true;
        msg!("Submitted result hash: {}", result_hash);
        Ok(())
    }

    pub fn approve_and_pay(ctx: Context<ApproveAndPay>) -> Result<()> {
        let bounty = &mut ctx.accounts.bounty;
        require!(bounty.solved, PowError::NotSolved);
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
                ],
            )?;
        }
        Ok(())
    }
}

#[derive(Accounts)]
#[instruction(id: u64)]
pub struct CreateBounty<'info> {
    #[account(
        init,
        payer = authority,
        seeds = [b"bounty", &id.to_le_bytes()],
        bump,
        space = Bounty::SPACE
    )]
    pub bounty: Account<'info, Bounty>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct SubmitWork<'info> {
    #[account(
        mut,
        seeds = [b"bounty", &bounty.id.to_le_bytes()],
        bump
    )]
    pub bounty: Account<'info, Bounty>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct ApproveAndPay<'info> {
    #[account(
        mut,
        seeds = [b"bounty", &bounty.id.to_le_bytes()],
        bump
    )]
    pub bounty: Account<'info, Bounty>,
    #[account(mut)]
    pub authority: Signer<'info>,
    #[account(mut)]
    pub solver: SystemAccount<'info>,
    pub system_program: Program<'info, System>,
}

#[account]
pub struct Bounty {
    pub id: u64,
    pub description: String,
    pub reward: u64,
    pub solved: bool,
}

impl Bounty {
    pub const MAX_DESCRIPTION: usize = 280;
    pub const SPACE: usize = 8 + 8 + 4 + Self::MAX_DESCRIPTION + 8 + 1;
}

#[error_code]
pub enum PowError {
    #[msg("Bounty not solved")]
    NotSolved,
}

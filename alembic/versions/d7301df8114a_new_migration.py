"""New Migration

Revision ID: d7301df8114a
Revises: 
Create Date: 2023-03-31 17:19:03.497152

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd7301df8114a'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('twitteraccount',
    sa.Column('id', sa.String(), autoincrement=False, nullable=False),
    sa.Column('username', sa.String(), nullable=True),
    sa.Column('time_created', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('time_updated', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_twitteraccount_id'), 'twitteraccount', ['id'], unique=False)
    op.create_table('tweetcluster',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('topic', sa.String(), nullable=True),
    sa.Column('key_words', sa.ARRAY(sa.String()), nullable=True),
    sa.Column('account_id', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['account_id'], ['twitteraccount.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_tweetcluster_id'), 'tweetcluster', ['id'], unique=False)
    op.create_table('tweet',
    sa.Column('id', sa.String(), autoincrement=False, nullable=False),
    sa.Column('data', sa.JSON(), nullable=True),
    sa.Column('cluster_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['cluster_id'], ['tweetcluster.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_tweet_id'), 'tweet', ['id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_tweet_id'), table_name='tweet')
    op.drop_table('tweet')
    op.drop_index(op.f('ix_tweetcluster_id'), table_name='tweetcluster')
    op.drop_table('tweetcluster')
    op.drop_index(op.f('ix_twitteraccount_id'), table_name='twitteraccount')
    op.drop_table('twitteraccount')
    # ### end Alembic commands ###
"""Add created_at field to Prediction model

Revision ID: d3c096b1185b
Revises: 6a11c8c5421a
Create Date: 2024-10-20 22:03:21.973936

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd3c096b1185b'
down_revision = '6a11c8c5421a'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('prediction', schema=None) as batch_op:
        batch_op.add_column(sa.Column('created_at', sa.DateTime(), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('prediction', schema=None) as batch_op:
        batch_op.drop_column('created_at')

    # ### end Alembic commands ###
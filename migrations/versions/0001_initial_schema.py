"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2026-10-14
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "documents",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("owner_sub", sa.String(128), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("s3_key", sa.String(512), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_documents_owner_sub", "documents", ["owner_sub"])

    op.create_table(
        "queries",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("session_id", sa.String(64), nullable=True),
        sa.Column("owner_sub", sa.String(128), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("outcome", sa.String(32), nullable=False),
        sa.Column("pipeline_version", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_queries_session_id", "queries", ["session_id"])
    op.create_index("ix_queries_owner_sub", "queries", ["owner_sub"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("query_id", sa.String(32), sa.ForeignKey("queries.id"), nullable=False),
        sa.Column("owner_sub", sa.String(128), nullable=False),
        sa.Column("rating", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_feedback_query_id", "feedback", ["query_id"])
    op.create_index("ix_feedback_owner_sub", "feedback", ["owner_sub"])

    op.create_table(
        "reports",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("document_id", sa.String(32), sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("owner_sub", sa.String(128), nullable=False),
        sa.Column("content", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_reports_document_id", "reports", ["document_id"])
    op.create_index("ix_reports_owner_sub", "reports", ["owner_sub"])


def downgrade():
    op.drop_index("ix_reports_owner_sub", table_name="reports")
    op.drop_index("ix_reports_document_id", table_name="reports")
    op.drop_table("reports")
    op.drop_index("ix_feedback_owner_sub", table_name="feedback")
    op.drop_index("ix_feedback_query_id", table_name="feedback")
    op.drop_table("feedback")
    op.drop_index("ix_queries_owner_sub", table_name="queries")
    op.drop_index("ix_queries_session_id", table_name="queries")
    op.drop_table("queries")
    op.drop_index("ix_documents_owner_sub", table_name="documents")
    op.drop_table("documents")

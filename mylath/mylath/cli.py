# mylath/cli.py
import click
from .storage.redis_storage import RedisStorage
from .api.graph_api import GraphAPI
from .config import MyLathConfig


@click.command()
@click.option('--host', default='localhost', help='API server host')
@click.option('--port', default=5000, help='API server port')
@click.option('--redis-host', default='localhost', help='Redis host')
@click.option('--redis-port', default=6379, help='Redis port')
@click.option('--redis-db', default=0, help='Redis database number')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def main(host, port, redis_host, redis_port, redis_db, debug):
    """Start MyLath API server"""
    click.echo(f"Starting MyLath server on {host}:{port}")
    click.echo(f"Redis connection: {redis_host}:{redis_port}/{redis_db}")
    
    try:
        storage = RedisStorage(
            host=redis_host,
            port=redis_port,
            db=redis_db
        )
        
        # Test Redis connection
        storage.redis.ping()
        click.echo("✓ Redis connection successful")
        
        api = GraphAPI(storage)
        click.echo("✓ API server initialized")
        
        api.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        click.echo(f"✗ Error starting server: {e}", err=True)
        exit(1)


if __name__ == '__main__':
    main()

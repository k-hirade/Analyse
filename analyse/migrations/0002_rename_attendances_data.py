# Generated by Django 3.2 on 2022-06-27 18:27

from django.conf import settings
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('analyse', '0001_initial'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Attendances',
            new_name='Data',
        ),
    ]

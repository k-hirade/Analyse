# Generated by Django 3.2 on 2022-06-27 20:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analyse', '0002_rename_attendances_data'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='data',
            options={'managed': True},
        ),
        migrations.AddField(
            model_name='data',
            name='datalist',
            field=models.CharField(max_length=15, null=True),
        ),
        migrations.AlterModelTable(
            name='data',
            table='datalist',
        ),
    ]
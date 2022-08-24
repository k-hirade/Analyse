from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Data(models.Model):
  user = models.ForeignKey(User, on_delete=models.CASCADE)
  datalist = models.CharField(max_length=15,null=True)
  class Meta:
    managed = True
    # migrationsの管理対象内とする
    db_table = 'datalist'

#  class Attendances(models.Model):
#      user = models.ForeignKey(User, on_delete=models.CASCADE)
#      attendance_time = models.DateTimeField(default=datetime.now)
#      leave_time = models.DateTimeField(null=True)

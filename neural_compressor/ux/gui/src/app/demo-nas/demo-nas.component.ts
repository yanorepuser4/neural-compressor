import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { ModelService } from '../services/model.service';
import { SocketService } from '../services/socket.service';

@Component({
  selector: 'app-demo-nas',
  templateUrl: './demo-nas.component.html',
  styleUrls: ['./demo-nas.component.scss', './../error/error.component.scss', './../optimizations/optimizations.component.scss']
})
export class DemoNASComponent implements OnInit {

  imageToShow: any;
  showProgressBar = false;
  executionData = { status: '' };

  constructor(
    private modelService: ModelService,
    private socketService: SocketService,
    public activatedRoute: ActivatedRoute,
  ) { }

  ngOnInit(): void {
    this.modelService.demoListReady$.subscribe(
      response => {
        this.setStatus();
      });

    this.activatedRoute.params.subscribe(
      params => {
        this.showProgressBar = true;
        this.imageToShow = null;
        this.setStatus();
      });

    this.socketService.executeNasFinish$.subscribe(
      (response: { status: string }) => {
        this.showProgressBar = false;
        this.executionData.status = response.status;
        this.getImageFromService();
      },
      error => {
        this.showProgressBar = false;
        this.modelService.openErrorDialog(error);
      });

    this.socketService.nasFinish$.subscribe(
      error => {
        this.modelService.openErrorDialog(error);
        this.showProgressBar = false;
      });
  }

  setStatus() {
    this.executionData = this.modelService.demoList.find(data => data.id === Number(this.activatedRoute.snapshot.params.id));
    if (this.executionData?.status === 'wip') {
      this.showProgressBar = true;
    } else {
      if (this.executionData?.status === 'success') {
        this.getImageFromService();
      }
      this.showProgressBar = false;
    }
  }

  runNAS() {
    this.showProgressBar = true;
    this.modelService.executeNAS(this.activatedRoute.snapshot.params.id)
      .subscribe(
        responseExecute => {
          this.modelService.listNAS()
            .subscribe(
              (response: { nas: any }) => {
                this.modelService.demoList = response.nas;
                this.modelService.demoListReady$.next(true);
              },
              error => {
                this.modelService.openErrorDialog(error);
              });
        },
        error => {
          this.modelService.openErrorDialog(error);
        }
      );
  }

  getImageFromService() {
    this.modelService.getImage(this.activatedRoute.snapshot.params.id)
      .subscribe(
        data => {
          this.createImageFromBlob(data);
          this.showProgressBar = false;
        },
        error => {
          this.showProgressBar = false;
        });
  }

  createImageFromBlob(image: Blob) {
    const reader = new FileReader();
    reader.addEventListener('load', () => {
      this.imageToShow = reader.result;
    }, false);

    if (image) {
      reader.readAsDataURL(image);
    }
  }

}

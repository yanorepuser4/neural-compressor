import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DemoNASComponent } from './demo-nas.component';

describe('DemoNASComponent', () => {
  let component: DemoNASComponent;
  let fixture: ComponentFixture<DemoNASComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ DemoNASComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DemoNASComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

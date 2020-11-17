<template>
  <v-container fluid>
    <v-card class="ma-3 pa-3">
      <v-card-title primary-title>
        <div class="headline primary--text">Create Dataset</div>
      </v-card-title>
      <v-card-text>
        <DatasetForm :dataset="newDataset"></DatasetForm>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn @click="cancel">Cancel</v-btn>
        <v-btn @click="reset">Reset</v-btn>
        <v-btn @click="submit"> Save </v-btn>
      </v-card-actions>
    </v-card>
  </v-container>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { Dataset, DatasetUpdate, DatasetCreate, DatasetFileType } from '@/api';
import { defaultDataset } from '@/interfaces';
import DatasetForm from '@/components/DatasetForm.vue';
import { dispatchCreateDataset } from '@/store/dataset/actions';
import { filterUndefined } from '@/utils';

@Component({ components: { DatasetForm } })
export default class CreateDataset extends Vue {
  public newDataset: DatasetCreate = defaultDataset();
  public valid = false;

  public async mounted() {
    this.reset();
  }

  public reset() {
    this.newDataset = defaultDataset();
    this.$validator.reset();
  }

  public cancel() {
    this.$router.back();
  }

  public async submit() {
    if (await this.$validator.validateAll()) {
      const filteredDataset: DatasetCreate = filterUndefined(this.newDataset);
      await dispatchCreateDataset(this.$store, filteredDataset);
      this.$router.push('/main/datasets');
    }
  }
}
</script>
